import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    """Load CSV data into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error(f"File {file_path} is empty.")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


def preprocess_data(df, key_columns, features):
    """Clean and preprocess the data."""
    df[key_columns] = df[key_columns].apply(pd.to_numeric, errors='coerce')
    df_clean = df.dropna(subset=key_columns)
    X = pd.get_dummies(df_clean[features], columns=['driving_style', 'tire_type'], drop_first=True)
    y = df_clean['trip_distance(km)']
    return X, y, df_clean


def plot_actual_vs_predicted(y_test, y_pred, title='Actual vs Predicted'):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Actual Trip Distance (km)')
    plt.ylabel('Predicted Trip Distance (km)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_range_comparison(df_clean, X_test_r, y_pred_r):
    """Plot comparison of EV max range estimates."""
    # Ensure the indices from X_test_r are aligned with df_clean and match the length of y_pred_r
    common_indices = df_clean.index.intersection(X_test_r.index)
    if len(common_indices) == len(y_pred_r):
        df_clean.loc[common_indices, 'predicted_range_rf'] = y_pred_r
    else:
        logging.error("Mismatch between the number of predictions and the number of indices to update.")
        return
    
    plt.figure(figsize=(12, 6))
    sample_df = df_clean.loc[common_indices].copy()
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