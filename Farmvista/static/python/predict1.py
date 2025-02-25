import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import joblib

# Load the saved model
model = joblib.load('BestModel.pkl')
file_path = '/content/Data_Set (Rice-India).xlsx'  # Correct path for Kaggle
data = pd.read_excel(file_path)

# Convert Date to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Sort data by Date
data = data.sort_values('Date')

# Additional Feature Engineering: Rolling means, Exponential Moving Average, etc.
data['Rolling_Mean'] = data['Rice Price (₹/kg)'].rolling(window=3).mean()
data['EMA'] = data['Rice Price (₹/kg)'].ewm(span=5).mean()

# Fill any NaN values created by rolling features
data.fillna(method='bfill', inplace=True)

# Create lag features
for lag in range(1, 13):  # Create lag features for the past 12 months
    data[f'Lag_{lag}'] = data['Rice Price (₹/kg)'].shift(lag)

# Drop rows with NaN values after shifting
data = data.dropna()

# Prepare features and target
X = data.drop(columns=["Rice Price (₹/kg)", "Date"])
y = data["Rice Price (₹/kg)"]

def future_predictions(model, days, data, X):
    # Get the last available date in the dataset
    last_date = data['Date'].max()
    
    # Prepare an empty DataFrame for the future dates
    future_df = pd.DataFrame()

    # Loop to iteratively predict the next 'days' number of days
    for i in range(1, days + 1):
        # Create a new row for the next day
        new_row = pd.DataFrame({'Date': [last_date + timedelta(days=i)]})
        
        # Copy the last known features
        for column in X.columns:
            if column.startswith('Lag_'):
                lag_number = int(column.split('_')[1])
                if lag_number >= i:
                    # Use the predicted value for this lag feature if available, otherwise use actual values
                    new_row[column] = future_df['Predicted Price (₹/kg)'].shift(i - 1).iloc[-lag_number + i - 1] if (i > 1 and -lag_number + i - 1 >= 0) else data[column].iloc[-lag_number] 
                else:
                    # Use the actual lag feature from the last day
                    new_row[column] = data[column].iloc[-lag_number]
            else:
                # Use the last known value for non-lag features and add some randomness to create more realistic variations
                new_row[column] = data[column].iloc[-1] * (1 + np.random.normal(0, 0.02)) # Adding 2% random noise
        
        # Append the new row to future_df
        future_df = pd.concat([future_df, new_row], ignore_index=True)

        # Drop the Date column for prediction
        X_future = future_df.drop(columns=['Date'])

        # Apply preprocessing (scaling) to the future data
        X_future_transformed = model.named_steps['preprocessor'].transform(X_future)

        # Predict the price for the new day
        future_df.loc[i-1, 'Predicted Price (₹/kg)'] = model.named_steps['regressor'].predict(X_future_transformed)[-1]

    return future_df

# Ask the user for the number of days to predict
days_to_predict = int(input("Enter the number of days to predict: "))

# Get future predictions
future_df = future_predictions(model, days_to_predict, data, X)

# Print the future predictions
print(future_df[['Date', 'Predicted Price (₹/kg)']])

# Plot the results for predicted days only
plt.figure(figsize=(14, 7))
plt.plot(future_df['Date'], future_df['Predicted Price (₹/kg)'], label=f'Predicted Future Price for {days_to_predict} Days', color='orange')
plt.title(f'Predicted Rice Prices for Next {days_to_predict} Days')
plt.xlabel('Date')
plt.ylabel('Rice Price (₹/kg)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()