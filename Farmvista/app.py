from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import joblib
import io
import base64
import os

app = Flask(__name__)

# Set a random seed for reproducibility
np.random.seed(42)

# Load the saved model and data
model = joblib.load('C:\\xampp\\htdocs\\Farmvista\\BestModel.pkl')
file_path = 'C:\\xampp\\htdocs\\Farmvista\\Data_Set (Rice-India).xlsx'  # Adjust this path if necessary
data = pd.read_excel(file_path)

# Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data['Rolling_Mean'] = data['Rice Price (₹/kg)'].rolling(window=3).mean()
data['EMA'] = data['Rice Price (₹/kg)'].ewm(span=5).mean()
data.fillna(method='bfill', inplace=True)

for lag in range(1, 13):
    data[f'Lag_{lag}'] = data['Rice Price (₹/kg)'].shift(lag)

data = data.dropna()
X = data.drop(columns=["Rice Price (₹/kg)", "Date"])
y = data["Rice Price (₹/kg)"]

def future_predictions(model, days, data, X, cache_file='predictions_cache.csv'):
    if os.path.exists(cache_file):
        cached_df = pd.read_csv(cache_file)
        if len(cached_df) == days:
            cached_df['Date'] = pd.to_datetime(cached_df['Date'])
            return cached_df

    last_date = data['Date'].max()
    future_df = pd.DataFrame()

    for i in range(1, days + 1):
        new_row = pd.DataFrame({'Date': [last_date + timedelta(days=i)]})
        
        for column in X.columns:
            if column.startswith('Lag_'):
                lag_number = int(column.split('_')[1])
                if lag_number >= i:
                    new_row[column] = (
                        future_df['Predicted Price (₹/kg)']
                        .shift(i - 1)
                        .iloc[-lag_number + i - 1]
                        if (i > 1 and -lag_number + i - 1 >= 0)
                        else data[column].iloc[-lag_number]
                    )
                else:
                    new_row[column] = data[column].iloc[-lag_number]
            else:
                new_row[column] = data[column].iloc[-1] * (1 + np.random.normal(0, 0.02))
        
        future_df = pd.concat([future_df, new_row], ignore_index=True)
        X_future = future_df.drop(columns=['Date'])
        X_future_transformed = model.named_steps['preprocessor'].transform(X_future)
        future_df.loc[i-1, 'Predicted Price (₹/kg)'] = model.named_steps['regressor'].predict(X_future_transformed)[-1]

    future_df.to_csv(cache_file, index=False)
    return future_df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        days_to_predict = int(request.form['number_of_days'])
        future_df = future_predictions(model, days_to_predict, data, X)

        # Create a plot
        plt.figure(figsize=(14, 7))
        plt.plot(future_df['Date'], future_df['Predicted Price (₹/kg)'], label=f'Predicted Future Price for {days_to_predict} Days', color='orange')
        plt.title(f'Predicted Rice Prices for Next {days_to_predict} Days')
        plt.xlabel('Date')
        plt.ylabel('Rice Price (₹/kg)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # Save plot to a BytesIO object and encode it to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rice Price Predictions</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f8f9fa;
                    color: #343a40;
                    text-align: center;
                    margin: 0;
                    padding: 20px;
                }
                h1 {
                    color: #007bff;
                    margin-bottom: 20px;
                }
                h2 {
                    color: #6c757d;
                }
                table {
                    margin: 20px auto;
                    border-collapse: collapse;
                    width: 80%;
                    background-color: #ffffff;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                }
                table th, table td {
                    border: 1px solid #dee2e6;
                    padding: 10px;
                    text-align: center;
                }
                table th {
                    background-color: #007bff;
                    color: #ffffff;
                }
                table tr:nth-child(even) {
                    background-color: #f8f9fa;
                }
                img {
                    margin: 20px auto;
                    max-width: 90%;
                    border: 2px solid #007bff;
                    border-radius: 5px;
                }
                a {
                    color: #007bff;
                    text-decoration: none;
                    font-size: 16px;
                }
                a:hover {
                    text-decoration: underline;
                }
                button {
                    background-color: #007bff;
                    color: #ffffff;
                    border: none;
                    padding: 10px 20px;
                    font-size: 16px;
                    border-radius: 5px;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <h1>Rice Price Predictions</h1>
            <h2>Future Predictions</h2>
            <table>
                <tr><th>Date</th><th>Predicted Price (₹/kg)</th></tr>
                {% for index, row in future_df.iterrows() %}
                <tr>
                    <td>{{ row['Date'].strftime('%Y-%m-%d') }}</td>
                    <td>{{ row['Predicted Price (₹/kg)'] | round(2) }}</td>
                </tr>
                {% endfor %}
            </table>
            <h2>Prediction Plot</h2>
            <img src="data:image/png;base64,{{ plot_url }}" alt="Prediction Plot" />
            <br>
            <a href="/"><button>Go Back</button></a>
        </body>
        </html>
        ''', future_df=future_df, plot_url=plot_url)

    return '''
        <form method="post">
            <label for="number_of_days">Number of Days</label>
            <input name="number_of_days" id="number_of_days" type="number" placeholder="Enter number of days" required>
            <label for="checkbox">Are you sure to continue?</label>
            <input id="checkbox" type="checkbox" required>
            <button type="submit">Submit</button>
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
