# Range_Prediction_EV

## Project Description

This project focuses on predicting the driving range of Electric Vehicles (EVs) based on various features like driving style, route characteristics, environmental conditions, and vehicle parameters. It explores different machine learning models to achieve this prediction.

## Datasets

The project utilizes several datasets, likely derived from the same source but processed differently:

*   `data_cleaned.csv`: A cleaned version of the original dataset.
*   `data_enc_label.csv`: Dataset with categorical features encoded using label encoding.
*   `data_enc_dummies.csv`: Dataset with categorical features encoded using one-hot encoding (dummies).
*   `volkswagen_e_golf.csv`: Specific dataset used for some models, potentially focused on a particular vehicle model.

*(Note: Ensure these data files are available or provide instructions on how to obtain them.)*

## Preprocessing

Data preprocessing is handled by the `preprocess/preprocess.py` script. Key steps include:
*   Handling missing values (imputation).
*   Scaling numerical features (StandardScaler).
*   Encoding categorical features (OneHotEncoder).
*   Feature selection (VarianceThreshold).
*   Outlier removal (IsolationForest).

## Models Implemented

Several regression models are implemented and evaluated:

1.  **Linear Regression:** (`linear_regression.py`) - Basic linear model, evaluated with and without feature scaling and using different encodings.
2.  **Decision Tree & Random Forest:** (`range_random_forest.py`) - Tree-based models, evaluated using different encodings.
3.  **Convolutional Neural Network (CNN):** (`range_cnn.py`) - A 1D CNN model implemented using TensorFlow/Keras.
4.  **Random Forest for Max Range Estimation:** (`range_main.py`) - Uses Random Forest to predict maximum range based on calculated estimates.

## Scripts

*   `preprocess/preprocess.py`: Contains the `Preprocessor` class for data cleaning and transformation.
*   `utils/utils.py`: Provides utility functions for loading data and plotting results (e.g., actual vs. predicted values).
*   `linear_regression.py`: Trains and evaluates Linear Regression models on different datasets.
*   `range_random_forest.py`: Trains and evaluates Decision Tree and Random Forest models.
*   `range_cnn.py`: Trains and evaluates a 1D CNN model.
*   `range_main.py`: Performs data loading, preprocessing, trains Linear Regression and Random Forest models (specifically for max range), and includes range estimation logic.

## Usage

To run the different model training scripts:

```bash
python Transormers/Range_Prediction_EV/linear_regression.py
python Transormers/Range_Prediction_EV/range_random_forest.py
python Transormers/Range_Prediction_EV/range_cnn.py
python Transormers/Range_Prediction_EV/range_main.py
```

*(Ensure the necessary datasets are located correctly relative to the scripts, e.g., in a `data` subdirectory or adjust the paths in the scripts/CONFIG sections.)*

## Dependencies

This project requires the following Python libraries:

*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   seaborn
*   tensorflow (for the CNN model)

It is recommended to use a virtual environment and install dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

You might want to create a `requirements.txt` file for easier dependency management.