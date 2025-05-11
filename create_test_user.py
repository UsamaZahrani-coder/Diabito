import os
from flask_insulin_api import app, db
from models import User

with app.app_context():
    try:
        test_user = User(username='test_user', email='test@example.com')
        test_user.set_password('Test12345678!')
        db.session.add(test_user)
        db.session.commit()
        print('Test user created successfully!')
    except Exception as e:
        print(f'Error creating test user: {e}')