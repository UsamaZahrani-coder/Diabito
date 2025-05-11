import traceback
from flask_insulin_api import app, db, User, CalculationHistory

with app.app_context():
    try:
        print("\nDEBUG: Testing calculation history endpoint...")
        user = User.query.first()
        if user:
            print(f"Found user: {user.username}")
            try:
                history_items = CalculationHistory.query.filter_by(user_id=user.id).all()
                print(f"Found {len(history_items)} history items")
                
                if history_items:
                    print("Testing to_dict() method on first item...")
                    item = history_items[0]
                    print("Item attributes:")
                    for attr in ['id', 'timestamp', 'glucose_value', 'target_glucose', 
                                'insulin_sensitivity', 'carb_insulin_ratio', 'carbs',
                                'sugar', 'protein', 'calories', 'fat', 'correction_dose',
                                'carb_dose', 'bolus_dose', 'bolus_dose_ceil', 'image_filename']:
                        print(f"  {attr}: {getattr(item, attr, 'Not found')}")
                    
                    try:
                        result = item.to_dict()
                        print("\nto_dict() result:")
                        for key, value in result.items():
                            print(f"  {key}: {value}")
                    except Exception as e:
                        print(f"\nERROR in to_dict(): {e}")
                        traceback.print_exc()
            except Exception as e:
                print(f"\nERROR with history query: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()