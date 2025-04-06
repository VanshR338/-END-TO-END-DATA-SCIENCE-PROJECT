# -END-TO-END-DATA-SCIENCE-PROJECT

NAME: Vansh Raina

INTERN ID: CT08WQF

DOMAIN: DATA SCIENCE DEVELOPMENT

DURATION: 4 WEEEKS

"MENTOR*: NEELA SANTOSH

# -Project Description:
The House Price Prediction project is a machine learning-powered web application designed to estimate the price of a house based on its features. With the growing interest in real estate analytics, predicting housing prices with accuracy and speed can significantly assist buyers, sellers, and real estate professionals in making informed decisions. This project combines data preprocessing, machine learning model training, and web deployment using Flask and FastAPI to provide real-time predictions via API endpoints.

The project begins by loading a dataset containing various attributes of houses, such as the number of bedrooms, bathrooms, area measurements, construction year, renovation year, geographic coordinates, and more. The dataset undergoes data preprocessing, which includes handling missing values and converting categorical variables into numerical format using one-hot encoding. After cleaning and encoding the data, a Random Forest Regressor model is trained on this dataset to learn patterns and relationships between the input features and the target variable — house price.

Once the model is trained and evaluated for performance (using metrics like Mean Squared Error), it is saved using Joblib for future use. Along with the model, the feature names used during training are also stored to ensure consistent input formatting during prediction.

# -For deployment, two APIs were developed:

Flask API: Provides a simple REST endpoint that accepts JSON data and returns the predicted house price.

FastAPI: Offers a modern, high-performance API with automatic Swagger documentation. It uses a Pydantic model to define and validate input data.

The FastAPI app features strong data validation using aliases, making it easier to map user-friendly field names with the internal model's expectations. The predict route accepts a POST request containing house features and returns the predicted price.

# -Project Components:
Dataset: CSV file containing house listings with features like area, location, number of rooms, etc.

Data Preprocessing Module:

Handling missing values

One-hot encoding of categorical variables

Feature selection

Model Training Script:

Train-test split

Random Forest model training

Model evaluation and saving

Feature Name Storage:

Saving feature columns used during training for consistent input structure

Flask API:

Simple REST endpoint for prediction

FastAPI Application:

Advanced API interface

Request validation using Pydantic

Swagger UI for testing

Testing Client:

Postman / Swagger UI for testing input and output

Model and File Storage:

Trained model (.pkl)

Feature names (.pkl)

# -Applications:
Real Estate Platforms: Integrate the API to provide instant price predictions to users browsing properties.

Financial Institutions: Use predictions for mortgage assessments or risk analysis.

Urban Planning: Estimate property values in development projects.

Educational Tools: Demonstrate machine learning applications to students and professionals.

Market Research: Analyze pricing trends by feeding batch inputs into the model.

# -OUTPUT
![Image](https://github.com/user-attachments/assets/0d052b9d-8aef-4ddc-8ab2-b01b9a524bbc)
![Image](https://github.com/user-attachments/assets/7f28edb4-3bea-46b7-a720-38e8a3560fee)
![Image](https://github.com/user-attachments/assets/b9413998-2423-4b4e-ba6e-620f318b5f2a)


