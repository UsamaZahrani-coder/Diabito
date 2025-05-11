# 🚀 Full Setup: Flask API + React Native Expo App

This guide provides a step-by-step process to set up a **Flask API** and connect it with a **React Native (Expo) App** for insulin calculations based on food image analysis.

---

## **📌 1. Install Python & Dependencies**

Make sure you have Python installed, then install the necessary libraries:

```sh
pip install flask flask-cors requests gunicorn
```

If you have a `requirements.txt` file, install everything with:

```sh
pip install -r requirements.txt
```

---

## **📌 2. Create the Flask API (`flask_insulin_api.py`)**

```python
import os
import math
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
API_USER_TOKEN = "YOUR_LOGMEAL_API_TOKEN"

@app.route("/", methods=["POST"])
def process_image():
    try:
        glucose_value = float(request.form["glucose_value"])
        target_glucose = float(request.form["target_glucose"])
        insulin_sensitivity = float(request.form["insulin_sensitivity"])
        carb_insulin_ratio = float(request.form["carb_insulin_ratio"])
        
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        image = request.files["image"]
        if image.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
        image.save(filepath)
        
        segmentation_url = "https://api.logmeal.com/v2/image/segmentation/complete"
        headers = {"Authorization": f"Bearer {API_USER_TOKEN}"}
        
        with open(filepath, "rb") as image_file:
            segmentation_response = requests.post(segmentation_url, files={"image": image_file}, headers=headers)
        segmentation_response.raise_for_status()
        
        nutrition_url = "https://api.logmeal.com/v2/recipe/nutritionalInfo"
        nutrition_payload = {"imageId": segmentation_response.json()["imageId"]}
        nutrition_response = requests.post(nutrition_url, json=nutrition_payload, headers=headers)
        nutrition_response.raise_for_status()
        
        nutrition_data = nutrition_response.json()
        total_nutrients = nutrition_data["nutritional_info"]["totalNutrients"]
        
        carbs = total_nutrients.get("CHOCDF", {}).get("quantity", 0)
        sugar = total_nutrients.get("SUGAR", {}).get("quantity", 0)
        protein = total_nutrients.get("PROCNT", {}).get("quantity", 0)
        
        correction_dose = (glucose_value - target_glucose) / insulin_sensitivity
        carb_dose = carbs / carb_insulin_ratio
        bolus_dose = correction_dose + carb_dose
        bolus_dose_ceil = math.ceil(bolus_dose)
        
        return jsonify({
            "carbs": carbs,
            "sugar": sugar,
            "protein": protein,
            "correction_dose": round(correction_dose, 2),
            "carb_dose": round(carb_dose, 2),
            "bolus_dose": round(bolus_dose, 2),
            "bolus_dose_ceil": bolus_dose_ceil
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

---

## **📌 3. Run Flask API Locally**

```sh
python flask_insulin_api.py
```

Once running, the API will be available at:
```
http://127.0.0.1:5000/
```

---

## **📌 4. Setup React Native (Expo) App**

Create a new Expo project:

```sh
npx create-expo-app insulin-calculator
cd insulin-calculator
npx expo install react-native-image-picker
npm install axios
npx expo start
```

---

## **📌 5. React Native Code (`App.js`)**

```javascript
import React, { useState } from 'react';
import { View, Text, TextInput, Button, Image, ActivityIndicator, Alert } from 'react-native';
import * as ImagePicker from 'react-native-image-picker';
import axios from 'axios';

const API_URL = "http://192.168.x.x:5000";  // Replace with your local network IP

export default function App() {
  const [glucose, setGlucose] = useState('');
  const [targetGlucose, setTargetGlucose] = useState('');
  const [insulinSensitivity, setInsulinSensitivity] = useState('');
  const [carbInsulinRatio, setCarbInsulinRatio] = useState('');
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const pickImage = async () => {
    ImagePicker.launchImageLibrary({ mediaType: 'photo' }, (response) => {
      if (!response.didCancel && !response.error) {
        setImage(response.assets[0].uri);
      }
    });
  };

  const calculateInsulin = async () => {
    if (!glucose || !targetGlucose || !insulinSensitivity || !carbInsulinRatio || !image) {
      Alert.alert('Error', 'Please fill all fields and upload an image');
      return;
    }
    
    setLoading(true);
    const formData = new FormData();
    formData.append('glucose_value', glucose);
    formData.append('target_glucose', targetGlucose);
    formData.append('insulin_sensitivity', insulinSensitivity);
    formData.append('carb_insulin_ratio', carbInsulinRatio);
    formData.append('image', {
      uri: image,
      type: 'image/jpeg',
      name: 'food.jpg',
    });
    
    try {
      const response = await axios.post(`${API_URL}/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (error) {
      Alert.alert('Error', 'Failed to process the image. Try again.');
    } finally {
      setLoading(false);
    }
  };
}
```

---

## **🚀 Final Steps**
- Start the Flask API
- Run `npx expo start`
- Test the app!