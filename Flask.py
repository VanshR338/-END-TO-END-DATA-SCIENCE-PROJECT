from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('house_price_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data
    json_data = request.get_json()
    input_data = pd.DataFrame(json_data, index=[0])
    
    # Predict
    prediction = model.predict(input_data)
    
    # Return result
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)