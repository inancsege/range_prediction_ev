#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from Transormers.Range_Prediction_EV.utils.utils import load_data, preprocess_data, plot_actual_vs_predicted

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    'file_path': os.getenv('CSV_FILE_PATH', '../../Range_Prediction_EV/volkswagen_e_golf.csv'),
    'test_size': 0.2,
    'random_state': 42,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001
}


def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=CONFIG['learning_rate']), loss='mse')
    return model


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

    # Reshape data for CNN
    X_reshaped = data_handler.X.values.reshape((data_handler.X.shape[0], data_handler.X.shape[1], 1))

    # Model training
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, data_handler.y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'])
    cnn_model = build_cnn_model((X_train.shape[1], 1))
    cnn_model.fit(X_train, y_train, epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'], verbose=1)

    # Evaluation
    y_pred = cnn_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logging.info(f"CNN - MAE: {mae}, RMSE: {rmse}, R2: {r2}")
    plot_actual_vs_predicted(y_test, y_pred)


if __name__ == "__main__":
    main() 