#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from Transormers.Range_Prediction_EV.utils.utils import load_data, preprocess_data, plot_actual_vs_predicted, plot_range_comparison
from Transormers.Range_Prediction_EV.preprocess.preprocess import preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    'file_path': os.getenv('CSV_FILE_PATH', '../../Range_Prediction_EV/volkswagen_e_golf.csv'),
    'test_size': 0.2,
    'random_state': 42
}


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_clean = None
        self.X = None
        self.y = None

    def load_data(self):
        """Load CSV data into a DataFrame."""
        try:
            self.df = pd.read_csv(self.file_path)
            logging.info(f"Data loaded successfully from {self.file_path}")
        except FileNotFoundError:
            logging.error(f"File not found: {self.file_path}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")

    def preprocess_data(self):
        """Clean and preprocess the data."""
        if self.df is not None:
            self.df['trip_distance(km)'] = pd.to_numeric(self.df['trip_distance(km)'], errors='coerce')
            key_columns = ['trip_distance(km)', 'quantity(kWh)', 'avg_speed(km/h)']
            self.df_clean = self.df.dropna(subset=key_columns)
            features = [
                'quantity(kWh)', 'power(kW)', 'consumption(kWh/100km)', 'avg_speed(km/h)',
                'city', 'motor_way', 'country_roads', 'A/C', 'park_heating',
                'ecr_deviation', 'driving_style', 'tire_type'
            ]
            self.X = pd.get_dummies(self.df_clean[features], columns=['driving_style', 'tire_type'], drop_first=True)
            self.y = self.df_clean['trip_distance(km)']


class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_linear_regression(self):
        """Train a Linear Regression model and evaluate it."""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'])
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Linear Regression - MAE: {mae}, RMSE: {rmse}, R2: {r2}")
        return y_test, y_pred

    def train_random_forest(self, target_range):
        """Train a Random Forest Regressor and evaluate it."""
        X_train, X_test, y_train, y_test = train_test_split(self.X, target_range, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'])
        rf_model = RandomForestRegressor(random_state=CONFIG['random_state'])
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Random Forest - MAE: {mae}, RMSE: {rmse}, R2: {r2}")
        return y_test, y_pred


class RangeEstimator:
    def __init__(self, df_clean):
        self.df_clean = df_clean

    def calculate_max_range(self):
        """Calculate the maximum battery capacity observed in the dataset."""
        max_capacity_kwh = self.df_clean['quantity(kWh)'].max()
        self.df_clean['max_range_estimate_km'] = (100 * max_capacity_kwh) / self.df_clean['consumption(kWh/100km)']
        logging.info(f"Max Capacity (kWh): {max_capacity_kwh}")
        return max_capacity_kwh

    def plot_range_comparison(self, X_test_r, y_pred_r):
        """Plot comparison of EV max range estimates."""
        self.df_clean.loc[X_test_r.index, 'predicted_range_rf'] = y_pred_r
        plt.figure(figsize=(12, 6))
        sample_df = self.df_clean.loc[X_test_r.index].copy()
        sample_df = sample_df.sort_values(by='fixed_range_35_8_km').reset_index(drop=True)
        plt.plot(sample_df['fixed_range_35_8_km'], label='Fixed Capacity (35.8 kWh)', linestyle='--')
        plt.plot(sample_df['max_range_estimate_km'], label='Max Observed Capacity (84.6 kWh)', linestyle='-.')
        plt.plot(sample_df['predicted_range_rf'], label='Predicted Range (Model)', linewidth=2)
        plt.title('Comparison of EV Max Range Estimates')
        plt.xlabel('Sample Index')
        plt.ylabel('Estimated Max Range (km)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    # Data handling
    data_handler = DataHandler(CONFIG['file_path'])
    data_handler.df = load_data(CONFIG['file_path'])
    data_handler.X, data_handler.y, data_handler.df_clean = preprocess_data(
        data_handler.df,
        key_columns=['trip_distance(km)', 'quantity(kWh)', 'avg_speed(km/h)'],
        features=[
            'quantity(kWh)', 'power(kW)', 'consumption(kWh/100km)', 'avg_speed(km/h)',
            'city', 'motor_way', 'country_roads', 'A/C', 'park_heating',
            'ecr_deviation', 'driving_style', 'tire_type'
        ]
    )

    # Model training
    model_trainer = ModelTrainer(data_handler.X, data_handler.y)
    y_test, y_pred = model_trainer.train_linear_regression()
    plot_actual_vs_predicted(y_test, y_pred)

    # Range estimation
    range_estimator = RangeEstimator(data_handler.df_clean)
    max_capacity_kwh = range_estimator.calculate_max_range()
    target_range = data_handler.df_clean['max_range_estimate_km']
    y_test_r, y_pred_r = model_trainer.train_random_forest(target_range)
    plot_range_comparison(data_handler.df_clean, data_handler.X, y_pred_r)


if __name__ == "__main__":
    main()


# In[2]:


# The following code is commented out because 'X' and 'y' are not defined in this context.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# lr_model = LinearRegression()
# lr_model.fit(X_train, y_train)
# y_pred = lr_model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
# (mae, rmse, r2)


# In[3]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

(mae, rmse, r2)


# In[4]:


import matplotlib.pyplot as plt

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual Trip Distance (km)')
plt.ylabel('Predicted Trip Distance (km)')
plt.title('Actual vs Predicted Trip Distance (Linear Regression)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[5]:


# Calculate the maximum battery capacity observed in the dataset
max_capacity_kwh = df_clean['quantity(kWh)'].max()

# Compute estimated maximum range for each record using the max capacity
df_clean['max_range_estimate_km'] = (100 * max_capacity_kwh) / df_clean['consumption(kWh/100km)']

max_capacity_kwh


# In[6]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create a new target for model training: max_range_estimate_km
target_range = df_clean['max_range_estimate_km']

# Reuse and update feature set with encoded categorical variables
features_for_range = df_clean[[
    'quantity(kWh)', 'power(kW)', 'consumption(kWh/100km)', 'avg_speed(km/h)',
    'city', 'motor_way', 'country_roads', 'A/C', 'park_heating', 'ecr_deviation',
    'driving_style', 'tire_type'
]]

# One-hot encode categorical features
features_encoded = pd.get_dummies(features_for_range, columns=['driving_style', 'tire_type'], drop_first=True)

# Split data
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(features_encoded, target_range, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_r, y_train_r)

# Predict and evaluate
y_pred_r = rf_model.predict(X_test_r)
mae_r = mean_absolute_error(y_test_r, y_pred_r)
rmse_r = mean_squared_error(y_test_r, y_pred_r)
r2_r = r2_score(y_test_r, y_pred_r)

(mae_r, rmse_r, r2_r)


# In[13]:


# Calculate fixed battery range (35.8 kWh) for comparison
df_clean['fixed_range_35_8_km'] = (100 * 35.8) / df_clean['consumption(kWh/100km)']

# Add model predictions to the DataFrame
df_clean.loc[X_test_r.index, 'predicted_range_rf'] = y_pred_r

# Plot comparison
plt.figure(figsize=(12, 6))
sample_df = df_clean.loc[X_test_r.index].copy()
sample_df = sample_df.sort_values(by='fixed_range_35_8_km').reset_index(drop=True)

plt.plot(sample_df['fixed_range_35_8_km'], label='Fixed Capacity (35.8 kWh)', linestyle='--')
plt.plot(sample_df['max_range_estimate_km'], label='Max Observed Capacity (84.6 kWh)', linestyle='-.')
plt.plot(sample_df['predicted_range_rf'], label='Predicted Range (Model)', linewidth=2)

plt.title('Comparison of EV Max Range Estimates')
plt.xlabel('Sample Index')
plt.ylabel('Estimated Max Range (km)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[19]:


# Calculate fixed battery range (35.8 kWh) for comparison
df_clean['fixed_range_35_8_km'] = (100 * 35.8) / df_clean['consumption(kWh/100km)']

# Add model predictions to the DataFrame
df_clean.loc[X_test_r.index, 'predicted_range_rf'] = y_pred_r

# Plot comparison
plt.figure(figsize=(12, 6))
sample_df = df_clean.loc[X_test_r.index].copy()
sample_df = sample_df.sort_values(by='fixed_range_35_8_km').reset_index(drop=True)

plt.plot(sample_df['fixed_range_35_8_km'], label='Fixed Capacity (35.8 kWh)', linestyle='--')
plt.plot(sample_df['max_range_estimate_km'], label='Max Observed Capacity (84.6 kWh)', linestyle='-.')
plt.plot(sample_df['predicted_range_rf'], label='Predicted Range (Model)', linewidth=2)

plt.title('Comparison of EV Max Range Estimates')
plt.xlabel('Sample Index')
plt.ylabel('Estimated Max Range (km)')
plt.legend()
plt.grid(True)

# Reverse the x-axis
plt.gca().invert_xaxis()

plt.show()


# In[9]:


plt.figure(figsize=(12, 6))
sample_df = df_clean.loc[X_test_r.index].copy()
sample_df = sample_df.sort_values(by='fixed_range_35_8_km').reset_index(drop=True)

plt.plot(sample_df['fixed_range_35_8_km'], label='Fixed Capacity (35.8 kWh)', linestyle='--')
plt.plot(sample_df['max_range_estimate_km'], label='Max Observed Capacity (84.6 kWh)', linestyle='-.')
plt.plot(sample_df['predicted_range_rf'], label='Predicted Range (Model)', linewidth=2)

plt.title('Comparison of EV Max Range Estimates (Y-Axis Reversed)')
plt.xlabel('Sample Index')
plt.ylabel('Estimated Max Range (km)')
plt.legend()
plt.grid(True)

# Reverse Y-axis
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

