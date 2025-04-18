
# 🫀 Deep Neural Network on Heart Disease Dataset

This project demonstrates how to build and evaluate a Deep Neural Network (DNN) for binary classification using a heart disease dataset. The goal is to predict the presence of heart disease based on various clinical attributes.

## 📁 Dataset

The dataset used in this project is sourced from [KaggleHub](https://www.kaggle.com/datasets/yasserh/heart-disease-dataset). It contains medical information such as age, cholesterol level, blood pressure, and more.

### Features:
- `age`
- `sex`
- `cp` (chest pain type)
- `trestbps` (resting blood pressure)
- `chol` (cholesterol)
- `fbs` (fasting blood sugar)
- `restecg` (resting ECG results)
- `thalach` (maximum heart rate achieved)
- `exang` (exercise-induced angina)
- `oldpeak`
- `slope`
- `ca`
- `thal`

The target variable (`target`) indicates the presence (1) or absence (0) of heart disease.

## 🧠 Model Architecture

The model is a deep neural network built with TensorFlow/Keras, featuring:
- Input layer with 13 features
- 7 hidden layers, each with 15 units and ReLU activation
- Dropout (0.2) for regularization
- Output layer with sigmoid activation for binary classification

### Hyperparameters:
- Loss: Binary Crossentropy
- Optimizer: Adam (`learning_rate=0.001`)
- Batch Size: 32
- Epochs: 1000
- Validation Split: 10%

## 🛠️ Preprocessing Steps

1. **Data Loading**: Dataset is downloaded and read using `pandas`.
2. **Missing Values Handling**: Imputation using mean strategy.
3. **Feature Scaling**: `MinMaxScaler` is used to normalize features to the `[0, 1]` range.
4. **Train-Test Split**: Dataset is split (90% training, 10% testing) using `train_test_split`.

## 🚀 Training

The model is trained with GPU acceleration (`/device:GPU:0`) for better performance. A history object is returned containing training metrics.

## 📈 Evaluation

The model is evaluated on the test set using:
- **Binary Accuracy**
- **Loss (Binary Crossentropy)**

Results are printed at the end of training.

## 📌 Future Improvements

- Implement EarlyStopping and ModelCheckpoint for better training control
- Experiment with different architectures or regularization strategies
- Use cross-validation for more robust evaluation

## 🧾 Requirements

Make sure the following packages are installed:
```bash
pip install kagglehub pandas numpy scikit-learn tensorflow
```

## 📎 Reference

- Dataset: [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/heart-disease-dataset)
