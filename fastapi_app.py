import pandas as pd

# Load the dataset
data = pd.read_csv('house_prices.csv')

# Display basic information
print(data.head())
print(data.info())

# Handle missing values (example: fill with median)
data.fillna(data.median(), inplace=True)

# Encode categorical variables (example: one-hot encoding)
data = pd.get_dummies(data, drop_first=True)

# Save the preprocessed data
data.to_csv('preprocessed_data.csv', index=False)

import pandas as pd

# Load the dataset
data = pd.read_csv('house_prices.csv')

# Display basic information
print(data.head())
print(data.info())

# Handle missing values (example: fill with median)
data.fillna(data.median(), inplace=True)

# Encode categorical variables (example: one-hot encoding)
data = pd.get_dummies(data, drop_first=True)

# Save the preprocessed data
data.to_csv('preprocessed_data.csv', index=False)


from sklearn.model_selection import train_test_split

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Define features (X) and target (y)
X = data.drop('Price', axis=1)
y = data['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print(list(X.columns))

# Save the feature names the model was trained with
feature_names = list(X.columns)
joblib.dump(feature_names, 'feature_names.pkl')


# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model

joblib.dump(model, 'house_price_model.pkl')

from fastapi import FastAPI
from pydantic import create_model
import joblib
import pandas as pd

app = FastAPI()

# Load model and feature names
model = joblib.load('house_price_model.pkl')
feature_names = joblib.load("feature_names_corrected.pkl")

# Dynamically create the input model with all required fields as float
fields = {feature: (float, ...) for feature in feature_names}
InputData = create_model("InputData", **fields)

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input data into a DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Predict using the model
        prediction = model.predict(input_df)[0]
        return {"predicted_price": prediction}
    except Exception as e:
        return {"error": str(e)}
