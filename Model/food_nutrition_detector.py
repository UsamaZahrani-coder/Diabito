import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import cv2
from tqdm import tqdm
import argparse
import seaborn as sns
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set up command line arguments
parser = argparse.ArgumentParser(description='Food Nutrition Detection from Images')
parser.add_argument('--data_dir', type=str, default='nutrition5k_dataset', help='Path to the Nutrition5k dataset')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--model', type=str, default='efficientnet_b4', choices=['resnet50', 'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b4'], help='Model architecture')
parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'evaluate'], help='Mode: train, predict, or evaluate')
parser.add_argument('--image_path', type=str, help='Path to image for prediction')
parser.add_argument('--use_all_data', action='store_true', help='Use all available data for training')
args = parser.parse_args()

# Define paths
# Check if we need to add an extra 'nutrition5k_dataset' to the path
# This handles the case where the data is in a subdirectory with the same name
if os.path.exists(os.path.join(args.data_dir, 'nutrition5k_dataset')):
    base_dir = os.path.join(args.data_dir, 'nutrition5k_dataset')
else:
    base_dir = args.data_dir

METADATA_DIR = os.path.join(base_dir, 'metadata')
IMAGERY_DIR = os.path.join(base_dir, 'imagery')
OVERHEAD_DIR = os.path.join(IMAGERY_DIR, 'realsense_overhead')

# Define the nutritional properties we want to predict
NUTRITION_PROPERTIES = ['mass', 'calories', 'fat', 'carb', 'protein']

# Custom dataset class for Nutrition5k
class Nutrition5kDataset(Dataset):
    def __init__(self, dish_ids, metadata_df, root_dir, transform=None):
        self.dish_ids = dish_ids
        self.metadata_df = metadata_df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.dish_ids)
    
    def __getitem__(self, idx):
        dish_id = self.dish_ids[idx]
        
        # Get the RGB image path
        img_path = os.path.join(self.root_dir, dish_id, 'rgb.png')
        
        # Check if the file exists before trying to open it
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            # Return a placeholder image and zeros for nutrition
            image = torch.zeros(3, 224, 224)
            nutrition = torch.zeros(len(NUTRITION_PROPERTIES))
            return image, nutrition
            
        # Load image
        try:
            # For PyTorch transforms
            if isinstance(self.transform, transforms.Compose):
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
            # For Albumentations transforms
            else:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.transform:
                    transformed = self.transform(image=image)
                    image = transformed["image"]
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and zeros for nutrition
            image = torch.zeros(3, 224, 224)
            nutrition = torch.zeros(len(NUTRITION_PROPERTIES))
            return image, nutrition
        
        # Get nutrition data
        try:
            # Get the row containing nutrition data for this dish
            nutrition_row = self.metadata_df[self.metadata_df.iloc[:, 0] == dish_id].iloc[0, 1:6]
            
            # Convert all values to float explicitly to handle potential string values
            nutrition_data = [float(val) if val and not pd.isna(val) else 0.0 for val in nutrition_row.values]
            nutrition = torch.tensor(nutrition_data, dtype=torch.float32)
        except (ValueError, TypeError) as e:
            print(f"Error processing nutrition data for dish {dish_id}: {e}")
            # Return zeros for nutrition if there's an error
            nutrition = torch.zeros(len(NUTRITION_PROPERTIES), dtype=torch.float32)
        
        return image, nutrition

# Define the CNN model with multiple architecture options and specialized carb prediction
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

# Function to load and preprocess data
def load_data(use_all_data=False):
    print("Loading data...")
    
    # Load metadata with error handling for inconsistent CSV formatting
    try:
        # Use more robust CSV parsing options to handle inconsistent number of fields
        cafe1_metadata = pd.read_csv(os.path.join(METADATA_DIR, 'dish_metadata_cafe1.csv'), 
                                    header=None, 
                                    engine='python',  # Use the python engine which is more flexible
                                    on_bad_lines='skip',  # Skip problematic lines
                                    quoting=3)  # QUOTE_NONE to avoid issues with quotes
        
        cafe2_metadata = pd.read_csv(os.path.join(METADATA_DIR, 'dish_metadata_cafe2.csv'), 
                                    header=None, 
                                    engine='python', 
                                    on_bad_lines='skip', 
                                    quoting=3)
        
        metadata_df = pd.concat([cafe1_metadata, cafe2_metadata], ignore_index=True)
        print(f"Successfully loaded metadata with {len(metadata_df)} entries")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        print("Attempting alternative parsing method...")
        
        # Alternative approach: read as text and parse manually
        try:
            with open(os.path.join(METADATA_DIR, 'dish_metadata_cafe1.csv'), 'r') as f:
                cafe1_lines = f.readlines()
            
            with open(os.path.join(METADATA_DIR, 'dish_metadata_cafe2.csv'), 'r') as f:
                cafe2_lines = f.readlines()
                
            # Process each line to extract the first 6 columns (dish_id and nutrition values)
            data = []
            for line in cafe1_lines + cafe2_lines:
                parts = line.strip().split(',')
                if len(parts) >= 6:  # Ensure we have at least the required columns
                    data.append(parts[:6])  # Take only the first 6 columns
            
            # Create DataFrame with the parsed data
            metadata_df = pd.DataFrame(data)
            print(f"Successfully parsed metadata manually with {len(metadata_df)} entries")
        except Exception as e:
            print(f"Failed to parse metadata: {e}")
            metadata_df = pd.DataFrame(columns=range(6))  # Empty DataFrame with 6 columns
    
    # Get all dish IDs that have overhead images
    all_dish_ids = [d for d in os.listdir(OVERHEAD_DIR) if os.path.isdir(os.path.join(OVERHEAD_DIR, d))]
    
    # Filter dish IDs to only include those with metadata and valid rgb.png files
    valid_dish_ids = []
    for dish_id in all_dish_ids:
        # Check if the dish has metadata
        if len(metadata_df[metadata_df.iloc[:, 0] == dish_id]) > 0:
            # Check if the rgb.png file exists
            rgb_path = os.path.join(OVERHEAD_DIR, dish_id, 'rgb.png')
            if os.path.exists(rgb_path):
                valid_dish_ids.append(dish_id)
            else:
                print(f"Skipping dish {dish_id}: rgb.png file not found at {rgb_path}")
    
    print(f"Found {len(valid_dish_ids)} valid dishes with both images and metadata")
    
    # Define image transformations with extensive augmentations for better generalization
    # Using Albumentations for more advanced augmentations
    train_transform = A.Compose([
        A.Resize(height=320, width=320),  # Larger initial size for more detail
        A.RandomCrop(height=288, width=288),  # Larger crop size
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=2, min_height=8, min_width=8, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(height=288, width=288),  # Match the training crop size
        A.CenterCrop(height=288, width=288),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    if use_all_data:
        # Use all data for training (no validation split)
        print("Using all available data for training...")
        train_dataset = Nutrition5kDataset(valid_dish_ids, metadata_df, OVERHEAD_DIR, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        # Create a small validation set (10% of data) just for tracking progress
        val_ids = valid_dish_ids[:int(len(valid_dish_ids) * 0.1)]
        val_dataset = Nutrition5kDataset(val_ids, metadata_df, OVERHEAD_DIR, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        # Split data into train and validation sets (80/20 split)
        train_ids, val_ids = train_test_split(valid_dish_ids, test_size=0.2, random_state=42)
        # Create datasets
        train_dataset = Nutrition5kDataset(train_ids, metadata_df, OVERHEAD_DIR, transform=train_transform)
        val_dataset = Nutrition5kDataset(val_ids, metadata_df, OVERHEAD_DIR, transform=val_transform)
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, metadata_df

# Function to train the model with advanced training techniques
def train_model(train_loader, val_loader):
    print("Training model with advanced optimization...")
    
    # Initialize model with selected architecture
    model = NutritionModel(model_name=args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using model architecture: {args.model} on {device}")
    
    # Define custom loss function with weighted emphasis on carbohydrate prediction
    def combined_loss(pred, target, alpha=0.8, carb_weight=2.0):
        # Calculate MSE and L1 loss for each nutritional property
        mse_per_output = torch.mean((pred - target) ** 2, dim=0)  # MSE for each output
        l1_per_output = torch.mean(torch.abs(pred - target), dim=0)  # L1 for each output
        
        # Apply higher weight to carbohydrate loss (index 3 is carb)
        carb_idx = 3
        mse_per_output[carb_idx] *= carb_weight
        l1_per_output[carb_idx] *= carb_weight
        
        # Combine losses
        mse_loss = torch.mean(mse_per_output)
        l1_loss = torch.mean(l1_per_output)
        
        return alpha * mse_loss + (1 - alpha) * l1_loss
    
    # Use AdamW optimizer which has better weight decay implementation
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine annealing learning rate scheduler with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart period after each restart
        eta_min=args.lr/100  # Minimum learning rate
    )
    
    # Training loop with early stopping and gradient clipping
    best_val_loss = float('inf')
    best_r2_score = float('-inf')  # Track R² score for model selection
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    # Gradient scaler for mixed precision training if using CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for images, nutrition in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, nutrition = images.to(device), nutrition.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision training if using CUDA
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = combined_loss(outputs, nutrition)
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                # Clip gradients to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training for CPU
                outputs = model(images)
                loss = combined_loss(outputs, nutrition)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for images, nutrition in tqdm(val_loader, desc="Validation"):
                images, nutrition = images.to(device), nutrition.to(device)
                
                outputs = model(images)
                loss = combined_loss(outputs, nutrition)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Collect predictions and targets for R² calculation
                all_targets.append(nutrition.cpu().numpy())
                all_predictions.append(outputs.cpu().numpy())
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Calculate R² score for validation set
        all_targets = np.vstack(all_targets)
        all_predictions = np.vstack(all_predictions)
        r2_scores = []
        
        # Calculate R² for each nutrition property
        for i in range(len(NUTRITION_PROPERTIES)):
            r2 = r2_score(all_targets[:, i], all_predictions[:, i])
            r2_scores.append(r2)
        
        # Average R² score across all properties
        avg_r2_score = np.mean(r2_scores)
        val_r2_scores.append(avg_r2_score)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val R²: {avg_r2_score:.4f}, LR: {current_lr:.6f}")
        
        # Save model based on R² score (higher is better)
        if avg_r2_score > best_r2_score:
            best_r2_score = avg_r2_score
            torch.save(model.state_dict(), 'best_nutrition_model.pth')
            print(f"Saved best model with R² score: {best_r2_score:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"R² score did not improve. Patience: {patience_counter}/{args.patience}")
        
        # Early stopping based on R² score
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training and validation loss
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot R² scores
    plt.subplot(1, 2, 2)
    plt.plot(val_r2_scores, label='Validation R² Score', color='green')
    plt.axhline(y=0.7, color='r', linestyle='--', label='Target R² (0.7)')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Validation R² Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.close()
    
    print(f"Training complete! Best validation R² score: {best_r2_score:.4f}")
    return model
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
    print("Training complete!")
    return model

# Function to predict nutrition from a single image
def predict_nutrition(image_path, metadata_df):
    print(f"Predicting nutrition for image: {image_path}")
    
    # Load the trained model with the same architecture used for training
    model = NutritionModel(model_name=args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model.load_state_dict(torch.load('best_nutrition_model.pth', map_location=device))
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first.")
        return
    
    model = model.to(device)
    model.eval()
    
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Make prediction
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Convert prediction to numpy array
    prediction = prediction.cpu().numpy()[0]
    
    # Display results
    print("\nPredicted Nutrition:")
    for i, prop in enumerate(NUTRITION_PROPERTIES):
        print(f"{prop.capitalize()}: {prediction[i]:.2f}")
    
    # Display the image with predictions
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(image))
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(NUTRITION_PROPERTIES, prediction)
    plt.title("Predicted Nutrition Values")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig('nutrition_prediction.png')
    plt.show()

# Function to evaluate model performance
def evaluate_model():
    print("Evaluating model performance...")
    
    # Load data
    _, val_loader, _ = load_data()
    
    # Load the trained model with the same architecture used for training
    model = NutritionModel(model_name=args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model.load_state_dict(torch.load('best_nutrition_model.pth', map_location=device))
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first.")
        return
    
    model = model.to(device)
    model.eval()
    
    # Initialize lists to store actual and predicted values for each property
    all_actual = {prop: [] for prop in NUTRITION_PROPERTIES}
    all_predicted = {prop: [] for prop in NUTRITION_PROPERTIES}
    
    # Evaluate on validation set
    with torch.no_grad():
        for images, nutrition in tqdm(val_loader, desc="Evaluating"):
            images, nutrition = images.to(device), nutrition.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Store actual and predicted values
            for i, prop in enumerate(NUTRITION_PROPERTIES):
                all_actual[prop].extend(nutrition[:, i].cpu().numpy())
                all_predicted[prop].extend(outputs[:, i].cpu().numpy())
    
    # Calculate metrics for each property
    print("\nModel Accuracy Metrics:")
    print("-" * 50)
    print(f"{'Property':<10} {'MAE':<10} {'MSE':<10} {'RMSE':<10} {'R²':<10}")
    print("-" * 50)
    
    overall_mae = 0
    overall_mse = 0
    overall_r2 = 0
    
    # Create a figure for scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, prop in enumerate(NUTRITION_PROPERTIES):
        actual = np.array(all_actual[prop])
        predicted = np.array(all_predicted[prop])
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        
        # Accumulate for overall metrics
        overall_mae += mae
        overall_mse += mse
        overall_r2 += r2
        
        # Print metrics
        print(f"{prop.capitalize():<10} {mae:<10.4f} {mse:<10.4f} {rmse:<10.4f} {r2:<10.4f}")
        
        # Plot actual vs predicted
        ax = axes[i]
        ax.scatter(actual, predicted, alpha=0.5)
        ax.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
        ax.set_title(f"{prop.capitalize()} (R² = {r2:.4f})")
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        
        # Add text with metrics
        text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Calculate and print overall metrics
    overall_mae /= len(NUTRITION_PROPERTIES)
    overall_mse /= len(NUTRITION_PROPERTIES)
    overall_r2 /= len(NUTRITION_PROPERTIES)
    print("-" * 50)
    print(f"{'Overall':<10} {overall_mae:<10.4f} {overall_mse:<10.4f} {np.sqrt(overall_mse):<10.4f} {overall_r2:<10.4f}")
    
    # Remove any unused subplot
    if len(NUTRITION_PROPERTIES) < len(axes):
        for j in range(len(NUTRITION_PROPERTIES), len(axes)):
            fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    print("\nEvaluation plots saved to 'model_evaluation.png'")
    
    # Create error distribution plots
    plt.figure(figsize=(15, 10))
    for i, prop in enumerate(NUTRITION_PROPERTIES):
        plt.subplot(2, 3, i+1)
        errors = np.array(all_predicted[prop]) - np.array(all_actual[prop])
        sns.histplot(errors, kde=True)
        plt.title(f"{prop.capitalize()} Error Distribution")
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    print("Error distribution plots saved to 'error_distribution.png'")
    plt.close('all')

# Main function
def main():
    if args.mode == 'train':
        # Use all data for training when requested via command line flag
        train_loader, val_loader, _ = load_data(use_all_data=args.use_all_data)
        print(f"Training with {'all available' if args.use_all_data else '80%'} data")
        train_model(train_loader, val_loader)
    elif args.mode == 'predict':
        if args.image_path is None:
            print("Error: Please provide an image path for prediction using --image_path")
            return
        _, _, metadata_df = load_data()
        predict_nutrition(args.image_path, metadata_df)
    elif args.mode == 'evaluate':
        evaluate_model()

if __name__ == "__main__":
    main()