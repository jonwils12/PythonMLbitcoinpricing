import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Define the Coinbase API endpoint for historical prices
endpoint = 'https://api.coinbase.com/v2/prices/BTC-USD/historic'

# Set the desired start and end dates for the historical data
start_date = '2022-01-01'
end_date = '2022-12-31'

# Build the request URL with the desired parameters
url = f'{endpoint}?start={start_date}&end={end_date}'

# Send the HTTP GET request to the Coinbase API
response = requests.get(url)

# Parse the response JSON data
data = response.json()['data']['prices']

# Create a DataFrame from the parsed data
df = pd.DataFrame(data, columns=['time', 'price'])

# Convert the time column to datetime format
df['time'] = pd.to_datetime(df['time'])

# Prepare the input features (X) and target variable (y) from the dataset
X = df[['time']]  # Example features: time
y = df['price']  # Example target variable: price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Use the trained model to make predictions on the testing set
y_pred = model.predict(X_test)

# Print the predicted prices and actual prices from the testing set
for pred, actual in zip(y_pred, y_test):
    print("Predicted price:", pred)
    print("Actual price:", actual)
    print()

# Use the trained model to make a prediction for a new set of input features
new_features = [[pd.to_datetime('2023-01-01')]]  # Example new features for prediction
predicted_price = model.predict(new_features)
print("Predicted price for new features:", predicted_price[0])
