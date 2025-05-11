from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Personal Information
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    weight = db.Column(db.Float)
    height = db.Column(db.Float)
    
    # Profile Picture
    profile_picture = db.Column(db.String(255))
    
    # Diabetes Settings
    target_glucose = db.Column(db.Float)
    insulin_sensitivity = db.Column(db.Float)
    carb_ratio = db.Column(db.Float)
    
    # Relationships
    calculation_history = db.relationship('CalculationHistory', backref='user', lazy=True, cascade='all, delete-orphan')

    def validate_password(self, password):
        if len(password) < 12:
            raise ValueError('Password must be at least 12 characters long')
        if not any(char.isdigit() for char in password):
            raise ValueError('Password must contain at least one number')

    def set_password(self, password):
        self.validate_password(password)
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'profile_picture': self.profile_picture,
            'weight': str(self.weight) if self.weight is not None else '',
            'height': str(self.height) if self.height is not None else '',
            'age': str(self.age) if self.age is not None else ''
        }

class CalculationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    glucose_value = db.Column(db.Float, nullable=False)
    target_glucose = db.Column(db.Float, nullable=False)
    insulin_sensitivity = db.Column(db.Float, nullable=False)
    carb_insulin_ratio = db.Column(db.Float, nullable=False)
    carbs = db.Column(db.Float, nullable=False)
    sugar = db.Column(db.Float, nullable=False)
    protein = db.Column(db.Float, nullable=False)
    calories = db.Column(db.Float, nullable=True)
    fat = db.Column(db.Float, nullable=True)
    correction_dose = db.Column(db.Float, nullable=False)
    carb_dose = db.Column(db.Float, nullable=False)
    bolus_dose = db.Column(db.Float, nullable=False)
    bolus_dose_ceil = db.Column(db.Integer, nullable=False)
    image_filename = db.Column(db.String(255))
    
    def to_dict(self):
        try:
            return {
                'id': self.id,
                'timestamp': self.timestamp.isoformat() if self.timestamp else '',
                'glucose_value': float(self.glucose_value) if self.glucose_value is not None else 0.0,
                'target_glucose': float(self.target_glucose) if self.target_glucose is not None else 0.0,
                'insulin_sensitivity': float(self.insulin_sensitivity) if self.insulin_sensitivity is not None else 0.0,
                'carb_insulin_ratio': float(self.carb_insulin_ratio) if self.carb_insulin_ratio is not None else 0.0,
                'carbs': float(self.carbs) if self.carbs is not None else 0.0,
                'sugar': float(self.sugar) if self.sugar is not None else 0.0,
                'protein': float(self.protein) if self.protein is not None else 0.0,
                'calories': float(self.calories) if self.calories is not None else 0.0,
                'fat': float(self.fat) if self.fat is not None else 0.0,
                'correction_dose': float(self.correction_dose) if self.correction_dose is not None else 0.0,
                'carb_dose': float(self.carb_dose) if self.carb_dose is not None else 0.0,
                'bolus_dose': float(self.bolus_dose) if self.bolus_dose is not None else 0.0,
                'bolus_dose_ceil': int(self.bolus_dose_ceil) if self.bolus_dose_ceil is not None else 0,
                'image_filename': self.image_filename if self.image_filename else ''
            }
        except Exception as e:
            # Provide a fallback in case of any conversion errors
            print(f"Error in CalculationHistory.to_dict(): {e}")
            return {
                'id': self.id,
                'timestamp': str(self.timestamp) if self.timestamp else '',
                'glucose_value': 0.0,
                'target_glucose': 0.0,
                'insulin_sensitivity': 0.0,
                'carb_insulin_ratio': 0.0,
                'carbs': 0.0,
                'sugar': 0.0,
                'protein': 0.0,
                'calories': 0.0,
                'fat': 0.0,
                'correction_dose': 0.0,
                'carb_dose': 0.0,
                'bolus_dose': 0.0,
                'bolus_dose_ceil': 0,
                'image_filename': ''
            }