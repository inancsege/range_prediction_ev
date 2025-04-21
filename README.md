# Range_Prediction_EV

## Overview

This project aims to predict the driving range of Electric Vehicles (EVs) using machine learning techniques. It analyzes various factors influencing EV range, such as driving behavior, route characteristics, and vehicle specifications, by training and evaluating several regression models.

## Project Structure

```
Range_Prediction_EV/
├── preprocess/
│   └── preprocess.py       # Data preprocessing logic (Preprocessor class)
├── utils/
│   └── utils.py            # Utility functions (data loading, plotting)
├── linear_regression.py    # Linear Regression model training and evaluation
├── range_cnn.py            # Convolutional Neural Network (CNN) model
├── range_main.py           # Main script coordinating LR/RF training and range estimation
├── range_random_forest.py  # Decision Tree and Random Forest model training
├── requirements.txt        # Project dependencies
├── README.md               # This file
└── data/                   # (Recommended) Directory for datasets
    ├── data_cleaned.csv
    ├── data_enc_label.csv
    ├── data_enc_dummies.csv
    └── volkswagen_e_golf.csv
```

## Data

The project uses several datasets:

*   `data_cleaned.csv`: Cleaned version of the primary dataset.
*   `data_enc_label.csv`: Dataset with label-encoded categorical features.
*   `data_enc_dummies.csv`: Dataset with one-hot encoded categorical features.
*   `volkswagen_e_golf.csv`: Specific dataset, possibly for the VW e-Golf model.

**Important:** The scripts assume data files are located relative to their position (e.g., `../../Range_Prediction_EV/data_cleaned.csv`). It is recommended to:
1.  Create a `data` directory within the `Transormers/Range_Prediction_EV` folder.
2.  Place all `.csv` files inside this `data` directory.
3.  Adjust the file paths within the Python scripts (e.g., in `load_data` calls or `CONFIG` dictionaries) to point to `data/your_dataset.csv`.

## Preprocessing (`preprocess/preprocess.py`)

The `Preprocessor` class handles data preparation, including:
*   Converting key columns to numeric types.
*   Dropping rows with missing essential data.
*   Imputing missing values (mean for numerical, most frequent for categorical).
*   Scaling numerical features using `StandardScaler`.
*   Encoding categorical features using `OneHotEncoder`.
*   Applying `VarianceThreshold` for basic feature selection.
*   Removing outliers using `IsolationForest`.

## Models & Scripts

*   **`linear_regression.py`**: Implements and evaluates `LinearRegression` from scikit-learn. Tests performance with and without feature scaling and using different data encodings (`data_cleaned.csv`, `data_enc_label.csv`, `data_enc_dummies.csv`).
*   **`range_random_forest.py`**: Trains and evaluates `DecisionTreeRegressor` and `RandomForestRegressor` models on the differently encoded datasets. Uses helper classes `DataLoader`, `RegressionModel`, and `Plotter`.
*   **`range_cnn.py`**: Defines and trains a 1D Convolutional Neural Network (CNN) using TensorFlow/Keras for range prediction, likely on the `volkswagen_e_golf.csv` dataset. Uses a `CONFIG` dictionary for hyperparameters.
*   **`range_main.py`**: Acts as a high-level script. It loads data (`volkswagen_e_golf.csv`), uses the `Preprocessor`, trains a `LinearRegression` model, and trains a `RandomForestRegressor` specifically for estimating maximum vehicle range based on calculated potential. Also includes plotting utilities. Uses a `CONFIG` dictionary.
*   **`utils/utils.py`**: Contains helper functions for common tasks like loading CSV files (`load_data`) and generating plots (`plot_actual_vs_predicted`, `plot_range_comparison`).

## Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>/Transormers/Range_Prediction_EV
    ```

2.  **Set up environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the virtual environment (example for bash/zsh)
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data:**
    *   Create a `data` directory: `mkdir data`
    *   Download or place your `.csv` datasets (`data_cleaned.csv`, `data_enc_label.csv`, `data_enc_dummies.csv`, `volkswagen_e_golf.csv`) into the `data` directory.
    *   **Verify and update file paths** within the Python scripts (`.py` files) to correctly reference the files in the `data/` directory (e.g., change `../../Range_Prediction_EV/data_cleaned.csv` to `data/data_cleaned.csv`).

5.  **Run Scripts:** Execute the desired model training script:
    ```bash
    python linear_regression.py
    python range_random_forest.py
    python range_cnn.py
    python range_main.py
    ```
    *(Note: Plots may be displayed during execution.)*

## Dependencies

All required Python libraries are listed in `requirements.txt`. Key dependencies include:
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `tensorflow`

Install them using `pip install -r requirements.txt`.