# ElevateLabs_Task3

##  Dataset Overview

The dataset contains various features of houses and their prices. The goal is to build a regression model that predicts the **price** of a house given its characteristics.

---

## Steps Performed

### 1. Import and Preprocess the Dataset
- Loaded the dataset using `pandas`.
- Handled any missing values (if present).
- Used **one-hot encoding** to convert categorical features into numerical.
  
### 2. Split Data into Train-Test Sets
- Separated features (`X`) from the target variable (`price`).
- Used `train_test_split()` to split the dataset into:
  - 80% training data
  - 20% test data

### 3. Fit a Linear Regression Model
- Used `LinearRegression` from `sklearn.linear_model` to train the model on the training set.

### 4. Evaluate the Model
- Evaluated the model on the test set using:
  - **MAE** (Mean Absolute Error)
  - **MSE** (Mean Squared Error)
  - **R² Score** (Coefficient of Determination)
- These metrics help assess how close predictions are to actual prices.

### 5. Visualize Predictions & Interpret Coefficients
- Plotted **Actual vs Predicted** prices using a scatter plot.
- Printed out **model coefficients**, which explain how each feature contributes to the predicted price.

---

##  Interpretation

- A good model will have **low MAE/MSE** and **R² close to 1.0**.
- The **scatterplot** helps visually check prediction performance — points close to the diagonal red line indicate better accuracy.
- **Coefficients** show the impact of each feature:
  - Positive value: increases the price
  - Negative value: decreases the price

---

## 🛠️ Tools & Libraries Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---
