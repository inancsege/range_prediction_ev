#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Transormers.Range_Prediction_EV.utils.utils import load_data, plot_actual_vs_predicted


# DataLoader class
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = load_data(file_path)

# RegressionModel class
class RegressionModel:
    def __init__(self, data, columns_x, column_y):
        self.data = data
        self.columns_x = columns_x
        self.column_y = column_y

    def train_model(self, model_type='decision_tree', **kwargs):
        """Train a model and return predictions and MSE."""
        if model_type == 'decision_tree':
            reg = DecisionTreeRegressor(random_state=kwargs.get('random_state', 1), max_depth=kwargs.get('max_depth'))
        elif model_type == 'random_forest':
            reg = RandomForestRegressor(random_state=kwargs.get('random_state', 1), n_estimators=kwargs.get('n_estimators', 100), max_depth=kwargs.get('max_depth'), n_jobs=-1)
        else:
            raise ValueError("Unsupported model type")
        return self._train_model(reg)

    def _train_model(self, reg):
        X = self.data[self.columns_x]
        Y = self.data[self.column_y]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return y_pred, mse, X_test, y_test

# Plotter class
class Plotter:
    @staticmethod
    def plot_predictions(X_test, y_test, y_pred, feature_name):
        """Plot actual vs predicted values."""
        plot_actual_vs_predicted(y_test, y_pred, title='Actual vs Predicted')

    @staticmethod
    def plot_prediction_difference(X_test, y_test, y_pred, feature_name):
        """Plot the difference between actual and predicted values."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_test[feature_name], y=y_test - y_pred, alpha=0.6)
        plt.title('Prediction Difference')
        plt.xlabel(feature_name)
        plt.ylabel('Difference (km)')
        plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    data_loader = DataLoader("../../Range_Prediction_EV/data_cleaned.csv")
    data_loader_l = DataLoader("../../Range_Prediction_EV/data_enc_label.csv")
    data_loader_d = DataLoader("../../Range_Prediction_EV/data_enc_dummies.csv")

    # Define columns
    columns_x = ['quantity(kWh)', 'city', 'motor_way', 'country_roads', 'consumption(kWh/100km)', 'A/C', 'park_heating', 'avg_speed(km/h)']
    column_y = 'trip_distance(km)'

    # Train and evaluate models
    model = RegressionModel(data_loader.data, columns_x, column_y)
    y_pred_tr, mse_tr, X_test_tr, y_test_tr = model.train_model(model_type='decision_tree', max_depth=10)
    Plotter.plot_predictions(X_test_tr, y_test_tr, y_pred_tr, 'quantity(kWh)')
    Plotter.plot_prediction_difference(X_test_tr, y_test_tr, y_pred_tr, 'quantity(kWh)')

    # Repeat for label encoded and dummy encoded data
    columns_xl = columns_x + ['tire_type_enc', 'driving_style_enc']
    model_l = RegressionModel(data_loader_l.data, columns_xl, column_y)
    y_predl_tr, msel_tr, X_testl_tr, y_testl_tr = model_l.train_model(model_type='decision_tree', max_depth=10)
    Plotter.plot_predictions(X_testl_tr, y_testl_tr, y_predl_tr, 'quantity(kWh)')
    Plotter.plot_prediction_difference(X_testl_tr, y_testl_tr, y_predl_tr, 'quantity(kWh)')

    columns_xd = columns_x + ['tire_type_Summer tires', 'tire_type_Winter tires', 'driving_style_Moderate', 'driving_style_Normal']
    model_d = RegressionModel(data_loader_d.data, columns_xd, column_y)
    y_predd_tr, msed_tr, X_testd_tr, y_testd_tr = model_d.train_model(model_type='decision_tree', max_depth=10)
    Plotter.plot_predictions(X_testd_tr, y_testd_tr, y_predd_tr, 'quantity(kWh)')
    Plotter.plot_prediction_difference(X_testd_tr, y_testd_tr, y_predd_tr, 'quantity(kWh)')

    # Print MSE results
    print(f"MSE for normal LR is {mse_tr}")
    print(f"MSE for label encoded LR is {msel_tr}")
    print(f"MSE for dummy encoded LR is {msed_tr}")

    # Random Forest
    y_pred_rf, mse_rf, X_test_rf, y_test_rf = model.train_model(model_type='random_forest', n_estimators=200, max_depth=10)
    Plotter.plot_predictions(X_test_rf, y_test_rf, y_pred_rf, 'quantity(kWh)')
    Plotter.plot_prediction_difference(X_test_rf, y_test_rf, y_pred_rf, 'quantity(kWh)')

    y_predl_rf, msel_rf, X_testl_rf, y_testl_rf = model_l.train_model(model_type='random_forest', n_estimators=200, max_depth=10)
    Plotter.plot_predictions(X_testl_rf, y_testl_rf, y_predl_rf, 'quantity(kWh)')
    Plotter.plot_prediction_difference(X_testl_rf, y_testl_rf, y_predl_rf, 'quantity(kWh)')

    y_predd_rf, msed_rf, X_testd_rf, y_testd_rf = model_d.train_model(model_type='random_forest', n_estimators=200, max_depth=10)
    Plotter.plot_predictions(X_testd_rf, y_testd_rf, y_predd_rf, 'quantity(kWh)')
    Plotter.plot_prediction_difference(X_testd_rf, y_testd_rf, y_predd_rf, 'quantity(kWh)')

    # Print MSE results for Random Forest
    print(f"MSE for normal RF is {mse_rf} km")
    print(f"MSE for label encoded RF is {msel_rf} km")
    print(f"MSE for dummy encoded RF is {msed_rf} km")


