#%%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error as mse

#%%
#%%

from Transormers.Range_Prediction_EV.utils.utils import load_data
from Transormers.Range_Prediction_EV.preprocess.preprocess import prepare_data, preprocess_data

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse_value = mse(y_test, y_pred)
    r2_value = lr.score(X_test, y_test)
    return y_test, y_pred, mse_value, r2_value


def train_and_evaluate_with_scaling(X, Y):
    # Check if X is sparse and set with_mean accordingly
    is_sparse = hasattr(X, "toarray")
    scaler = StandardScaler(with_mean=not is_sparse)
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, random_state=1)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse_value = mse(y_test, y_pred)
    r2_value = lr.score(X_test, y_test)
    cv_scores = cross_val_score(lr, X_scaled, Y, cv=5, scoring='neg_mean_squared_error')
    cv_mse = -cv_scores.mean()
    return y_test, y_pred, mse_value, r2_value, cv_mse


def main():
    # Load datasets
    data = load_data("../../Range_Prediction_EV/data_cleaned.csv")
    data_enc_l = load_data("../../Range_Prediction_EV/data_enc_label.csv")
    data_enc_dum = load_data("../../Range_Prediction_EV/data_enc_dummies.csv")

    if data is not None:
        X, Y, df_clean = preprocess_data(data, ['trip_distance(km)'], ['fuel_note','manufacturer', 'model', 'version', 'odometer',
           'trip_distance(km)', 'fuel_type','tire_type', 'driving_style'])
        y_test, y_pred, mse_value, r2_value, cv_mse = train_and_evaluate_with_scaling(X, Y)
        logging.info(f"MSE for normal LR: {mse_value}, R2: {r2_value}, CV MSE: {cv_mse}")
        plot_actual_vs_predicted(y_test, y_pred, "Normal")

    if data_enc_l is not None:
        X_enc_l, Y_enc_l, df_clean_enc_l = preprocess_data(data_enc_l, ['trip_distance(km)'], ['fuel_note','manufacturer', 'model', 'version', 'odometer',
           'trip_distance(km)', 'fuel_type','tire_type', 'driving_style'])
        y_test_enc_l, y_pred_enc_l, mse_enc_l, r2_enc_l, cv_mse_enc_l = train_and_evaluate_with_scaling(X_enc_l, Y_enc_l)
        logging.info(f"MSE for label encoded LR: {mse_enc_l}, R2: {r2_enc_l}, CV MSE: {cv_mse_enc_l}")
        plot_actual_vs_predicted(y_test_enc_l, y_pred_enc_l, "Label Encoded")

    if data_enc_dum is not None:
        X_enc_dum, Y_enc_dum, df_clean_enc_dum = preprocess_data(data_enc_dum, ['trip_distance(km)'], ['fuel_note','manufacturer', 'model', 'version', 'odometer',
           'trip_distance(km)', 'fuel_type'])
        y_test_enc_dum, y_pred_enc_dum, mse_enc_dum, r2_enc_dum, cv_mse_enc_dum = train_and_evaluate_with_scaling(X_enc_dum, Y_enc_dum)
        logging.info(f"MSE for dummy encoded LR: {mse_enc_dum}, R2: {r2_enc_dum}, CV MSE: {cv_mse_enc_dum}")
        plot_actual_vs_predicted(y_test_enc_dum, y_pred_enc_dum, "Dummy Encoded")


if __name__ == "__main__":
    main()

#%%
