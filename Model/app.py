import os
import torch
import numpy as np
from PIL import Image
# Set Matplotlib to use non-interactive Agg backend to avoid threading issues
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt
from torchvision import transforms, models
import torch.nn as nn
from flask import Flask, request, render_template, redirect, url_for, flash
import io
import base64
import timm

# Define the CNN model (same as in food_nutrition_detector.py)
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

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'nutrition_detection_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the nutritional properties we want to predict
NUTRITION_PROPERTIES = ['mass', 'calories', 'fat', 'carb', 'protein']

# Load the trained model
def load_model():
    # Use EfficientNet B4 as default model architecture (same as in training)
    model = NutritionModel(model_name='efficientnet_b4')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model.load_state_dict(torch.load('best_nutrition_model.pth', map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
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

# Function to create visualization
def create_visualization(image, prediction_results):
    plt.figure(figsize=(10, 6))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(image))
    plt.title("Food Image")
    plt.axis('off')
    
    # Display the nutrition prediction as a bar chart
    plt.subplot(1, 2, 2)
    bars = plt.bar(NUTRITION_PROPERTIES, [prediction_results[prop] for prop in NUTRITION_PROPERTIES])
    plt.title("Predicted Nutrition Values")
    plt.ylabel("Amount")
    plt.tight_layout()
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode the buffer to base64 for HTML display
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is available
    model, device = load_model()
    if model is None:
        flash('Error: Model not found. Please train the model first.')
        return redirect(url_for('index'))
    
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
        return redirect(url_for('index'))
    
    try:
        # Open and process the image
        image = Image.open(file.stream).convert('RGB')
        
        # Make prediction
        prediction_results = predict_nutrition(image, model, device)
        
        # Create visualization
        visualization = create_visualization(image, prediction_results)
        
        return render_template('result.html', 
                               prediction=prediction_results, 
                               visualization=visualization)
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)