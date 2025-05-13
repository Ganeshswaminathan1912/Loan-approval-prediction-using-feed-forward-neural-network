
# Loan Approval Prediction — README

## Overview

This project predicts loan approval status using a neural network.  
The model uses Keras and Keras Tuner for hyperparameter tuning (Random Search, Bayesian Optimization, Hyperband).



## Steps to Run the Colab Notebook

### Upload the Dataset

- Make sure loan_data.csv is available in your working directory (Colab or local Jupyter).
- You can upload it using:

python
from google.colab import files
files.upload()


Or directly upload through the Colab interface.



### Install Required Libraries

In the first cell, install all necessary packages:

python
!pip install keras-tuner


Other libraries required (pre-installed in Colab by default):

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow

If running locally, you can install them via:

bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras-tuner




### Run the Notebook Cells in Order

- **Preprocessing**: Encoding categorical variables, scaling numerical ones using ColumnTransformer and MinMaxScaler.
- **Model Building**: Neural network with tunable layers and optimizers.
- **Hyperparameter Tuning**: Using RandomSearch, BayesianOptimization, and Hyperband.
- **Model Evaluation**: Accuracy and Confusion Matrix for all tuning strategies.
- **Visualization**: Plotting training and validation metrics.



### Reproducing Results

To reproduce the results exactly:
- Set random_state=42 in all random functions (train_test_split and any random number generation).
- Use same max_trials and executions_per_trial in Keras Tuner search.
- Use early stopping callbacks (if applicable) for consistency.
- The model should achieve perfect accuracy (1.0) as seen in the provided notebook outputs (if data is unchanged).



### Notes

- This notebook is optimized for running on **Google Colab**.
- The training should complete within reasonable time with free-tier Colab GPU or CPU.
- Results may slightly vary if dataset is modified or hyperparameter search space is changed.

