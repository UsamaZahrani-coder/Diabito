import os
import math
import json
import uuid
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from models import db, User, CalculationHistory
import jwt
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import numpy as np
import timm
from torchvision import transforms, models

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
CORS(app)

db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or not all(k in data for k in ['username', 'email', 'password']):
        return jsonify({'error': 'Missing required fields'}), 400
        
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
        
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
        
    user = User()
    user.username = data['username']
    user.email = data['email']
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not all(k in data for k in ['username', 'password']):
        return jsonify({'error': 'Missing username or password'}), 400
        
    user = User.query.filter_by(username=data['username']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid username or password'}), 401
        
    token = jwt.encode(
        {
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=1)
        },
        app.config['SECRET_KEY'],
        algorithm='HS256'
    )
    
    return jsonify({
        'token': token,
        'user': user.to_dict()
    }), 200

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Define the CNN model for nutrition prediction
class NutritionModel(nn.Module):
    def __init__(self, model_name='efficientnet_b4', num_outputs=5):
        super(NutritionModel, self).__init__()
        
        if model_name == 'resnet50':
            # Use a pre-trained ResNet50 model
            self.base_model = models.resnet50(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name.startswith('efficientnet'):
            # Use a pre-trained EfficientNet model from timm
            self.base_model = timm.create_model(model_name, pretrained=True)
            num_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Feature normalization layer to help with numerical stability
        self.feature_norm = nn.LayerNorm(num_features)
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Specialized branch for carbohydrate prediction with additional layers
        self.carb_branch = nn.Sequential(
            nn.Linear(512, 512),  # Additional layer for carb prediction
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output just for carbs
        )
        
        # Branch for other nutritional properties
        self.other_nutrients_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs - 1)  # Output for all except carbs
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Apply better weight initialization for improved convergence
        for module in [self.shared_layers, self.carb_branch, self.other_nutrients_branch]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features from base model
        features = self.base_model(x)
        features = self.feature_norm(features)  # Normalize features for better stability
        
        # Pass through shared layers
        shared_features = self.shared_layers(features)
        
        # Get carb prediction from specialized branch
        carb_output = self.carb_branch(shared_features)
        
        # Get other nutrition predictions
        other_outputs = self.other_nutrients_branch(shared_features)
        
        # Concatenate outputs in the correct order: [mass, calories, fat, carb, protein]
        # Assuming the order in other_outputs is [mass, calories, fat, protein]
        carb_idx = 3  # Index where carb should be inserted
        outputs = torch.cat([
            other_outputs[:, :carb_idx],  # mass, calories, fat
            carb_output,                  # carb
            other_outputs[:, carb_idx:]   # protein
        ], dim=1)
        
        return outputs

# Define the nutritional properties we want to predict
NUTRITION_PROPERTIES = ['mass', 'calories', 'fat', 'carb', 'protein']

# Load the trained model
def load_nutrition_model():
    # Use EfficientNet B4 as default model architecture
    model = NutritionModel(model_name='efficientnet_b4')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model', 'best_nutrition_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return None, device

# Image transformation for prediction
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict nutrition from an image
def predict_nutrition(image, model, device):
    # Preprocess the image
    image_tensor = predict_transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Convert prediction to numpy array
    prediction = prediction.cpu().numpy()[0]
    
    # Create results dictionary
    results = {}
    for i, prop in enumerate(NUTRITION_PROPERTIES):
        results[prop] = round(float(prediction[i]), 2)
    
    return results

# Load the model at startup
nutrition_model, device = load_nutrition_model()

def get_token_auth_header():
    auth_header = request.headers.get('Authorization', None)
    if not auth_header:
        return None
    parts = auth_header.split()
    if parts[0].lower() != 'bearer' or len(parts) != 2:
        return None
    return parts[1]

@app.route('/get-profile', methods=['GET'])
def get_profile():
    token = get_token_auth_header()
    if not token:
        return jsonify({'error': 'Authorization header is required'}), 401
    
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user = User.query.get(payload['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        return jsonify(user.to_dict()), 200
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401

@app.route('/profile-picture', methods=['POST'])
def update_profile_picture():
    token = get_token_auth_header()
    if not token:
        return jsonify({'error': 'Authorization header is required'}), 401
        
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user = User.query.get(payload['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
    
        if 'profile_picture' not in request.files:
            return jsonify({'error': 'No profile picture provided'}), 400
            
        profile_pic = request.files['profile_picture']
        if not profile_pic or not profile_pic.filename:
            return jsonify({'error': 'No selected file'}), 400
    
        # Ensure filename is secure
        filename = secure_filename(profile_pic.filename)
        # Generate unique filename
        unique_filename = f"{str(uuid.uuid4())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Process and save image
        img = Image.open(BytesIO(profile_pic.read()))
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # Compress image
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Further compress if needed
        if len(img_byte_arr) > 300 * 1024:  # 300KB
            quality = int(85 * (300 * 1024 / len(img_byte_arr)))
            quality = max(30, min(quality, 85))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=quality)
            img_byte_arr = img_byte_arr.getvalue()
        
        # Save the compressed image
        with open(filepath, 'wb') as f:
            f.write(img_byte_arr)
        
        # Delete old profile picture if it exists
        if user.profile_picture:
            old_filepath = os.path.join(app.config['UPLOAD_FOLDER'], user.profile_picture)
            try:
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)
            except Exception as e:
                print(f"Error removing old profile picture: {e}")
    
        # Update user's profile picture URL
        user.profile_picture = unique_filename
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Profile picture updated successfully',
            'profile_picture': unique_filename
        }), 200
        
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/update-profile', methods=['PUT'])
def update_profile():
    token = get_token_auth_header()
    if not token:
        return jsonify({'error': 'Authorization header is required'}), 401
        
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user = User.query.get(payload['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Update user profile fields
        if 'weight' in data:
            user.weight = float(data['weight'])
        if 'height' in data:
            user.height = float(data['height'])
        if 'age' in data:
            user.age = int(data['age'])

        db.session.commit()
        return jsonify({'message': 'Profile updated successfully'}), 200
            
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
            
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route("/calculation-history", methods=["GET"])
def get_calculation_history():
    token = get_token_auth_header()
    if not token:
        return jsonify({"error": "Authorization header is required"}), 401
        
    try:
        payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        user = User.query.get(payload["user_id"])
        if not user:
            return jsonify({"error": "User not found"}), 404
            
        history_items = CalculationHistory.query.filter_by(user_id=user.id).order_by(CalculationHistory.timestamp.desc()).limit(10).all()
        return jsonify([item.to_dict() for item in history_items]), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token has expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/", methods=["POST"])
def process_image():
    try:
        # Check if form data exists
        if not request.form:
            return jsonify({"error": "No form data provided"}), 400

        # Validate and extract form data first
        try:
            glucose_value = float(request.form.get("glucose_value", ""))
            target_glucose = float(request.form.get("target_glucose", ""))
            insulin_sensitivity = float(request.form.get("insulin_sensitivity", ""))
            carb_insulin_ratio = float(request.form.get("carb_insulin_ratio", ""))
        except ValueError:
            return jsonify({"error": "Invalid numeric values in form data"}), 400

        # Check for image file after form data validation
        if not request.files or "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        image = request.files["image"]
        if image.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        # Save and compress image
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
        
        # Open and convert image to JPEG format using PIL
        img = Image.open(BytesIO(image.read()))
        
        # Convert to RGB if image is not in RGB mode
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
            
        # Get original format for error handling
        original_format = img.format or 'Unknown'
            
        # Calculate compression quality based on file size
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr = img_byte_arr.getvalue()
        
        # If image is still too large, compress further
        if len(img_byte_arr) > 300 * 1024:  # 300KB
            quality = int(85 * (300 * 1024 / len(img_byte_arr)))
            quality = max(30, min(quality, 85))  # Keep quality between 30 and 85
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=quality)
            img_byte_arr = img_byte_arr.getvalue()
        
        # Save the compressed image
        with open(filepath, 'wb') as f:
            f.write(img_byte_arr)
        
        try:
            # Use our custom nutrition model instead of LogMeal API
            if nutrition_model is None:
                return jsonify({"error": "Nutrition model not loaded. Please check the model file."}), 500
                
            # Open the image for prediction
            img_for_prediction = Image.open(BytesIO(img_byte_arr)).convert('RGB')
            
            # Get nutrition prediction from our model
            try:
                nutrition_results = predict_nutrition(img_for_prediction, nutrition_model, device)
            except Exception as e:
                return jsonify({"error": f"Error predicting nutrition: {str(e)}"}), 500
                
            # Extract the nutritional values we need
            carbs = nutrition_results.get('carb', 0)
            # We don't have sugar in our model output, so we'll estimate it as a percentage of carbs
            sugar = round(carbs * 0.3, 2)  # Assuming sugar is roughly 30% of carbs
            protein = nutrition_results.get('protein', 0)
            
            # Additional nutritional information that might be useful
            calories = nutrition_results.get('calories', 0)
            fat = nutrition_results.get('fat', 0)
        except Exception as e:
            return jsonify({"error": f"Error processing image with nutrition model: {str(e)}"}), 500
        
        correction_dose = (glucose_value - target_glucose) / insulin_sensitivity
        carb_dose = carbs / carb_insulin_ratio
        bolus_dose = correction_dose + carb_dose
        bolus_dose_ceil = math.ceil(bolus_dose)
        
        response_data = {
            "carbs": carbs,
            "sugar": sugar,
            "protein": protein,
            "calories": calories,
            "fat": fat,
            "correction_dose": round(correction_dose, 2),
            "carb_dose": round(carb_dose, 2),
            "bolus_dose": round(bolus_dose, 2),
            "bolus_dose_ceil": bolus_dose_ceil
        }
        
        # Save calculation to history if user is authenticated
        token = get_token_auth_header()
        if token:
            try:
                payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
                user = User.query.get(payload["user_id"])
                if user:
                    # Generate a unique filename for the image
                    unique_filename = f"{str(uuid.uuid4())}.jpeg"
                    permanent_filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
                    
                    # Save a copy of the image for history
                    with open(permanent_filepath, 'wb') as f:
                        f.write(img_byte_arr)
                    
                    # Create history record
                    history_item = CalculationHistory(
                        user_id=user.id,
                        glucose_value=glucose_value,
                        target_glucose=target_glucose,
                        insulin_sensitivity=insulin_sensitivity,
                        carb_insulin_ratio=carb_insulin_ratio,
                        carbs=carbs,
                        sugar=sugar,
                        protein=protein,
                        calories=calories if 'calories' in locals() else 0,
                        fat=fat if 'fat' in locals() else 0,
                        correction_dose=round(correction_dose, 2),
                        carb_dose=round(carb_dose, 2),
                        bolus_dose=round(bolus_dose, 2),
                        bolus_dose_ceil=bolus_dose_ceil,
                        image_filename=unique_filename
                    )
                    db.session.add(history_item)
                    db.session.commit()
            except Exception as e:
                print(f"Error saving calculation history: {e}")
        else:
            # Clean up the uploaded file if not saving to history
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Error removing temporary file: {e}")
            
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)