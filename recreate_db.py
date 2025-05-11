import os
from flask_insulin_api import app, db
from models import User

with app.app_context():
    # Drop existing tables
    db.drop_all()
    # Create new tables with updated schema
    db.create_all()
    print('Database recreated successfully!')