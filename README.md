# Flight Price Prediction using Machine Learning

## Author
Suyog Shrestha  
Data Science & Business @ Knox College - June 2027

---

## Project Overview
This project focuses on building a **machine learning regression model** to predict flight ticket prices based on various features such as airline, source, destination, duration, number of stops, and departure time.

The project follows an **end-to-end machine learning workflow**, including data preprocessing, exploratory data analysis, model training, evaluation, hyperparameter tuning, and model saving.

---

## How To Run This Project
1. Clone the repository
2. Install required dependencies (pip install -r requirements.txt
)
3. Open the Jupyter Notebook (jupyter notebook flight_price_prediction.ipynb)
4. Run all cells sequentially

---

## Exploratory Data Analysis (EDA)
The following analyses were performed:
- Price distribution using **histograms and box plots**
- Relationship between **flight duration and price**
- Airline-wise price comparison using **box plots**
- Effect of number of stops on ticket prices
- Departure time distribution analysis

EDA helped identify trends, outliers, and important predictors.

---

## Feature Engineering
- Applied **One-Hot Encoding** for categorical features such as `Source` and `Airline`
- Used **Label Encoding / mapping** for ordinal features like `Total_Stops`
- Converted time-based features into numerical formats
- Removed redundant and irrelevant columns
- Ensured all features were numerical for model training

---

## Models Trained
The following regression models were trained and evaluated:

| Model | R² Score | RMSE | MAE |
|------|---------|------|-----|
| **Random Forest Regressor** | **0.81** | **1915** | **1176** |
| Decision Tree Regressor | 0.68 | 2499 | 1399 |
| K-Nearest Neighbors | 0.63 | 2677 | 1750 |
| Linear Regression | 0.57 | 2890 | 2014 |

**Random Forest Regressor** achieved the best performance and was selected as the final model.

---

## Hyperparameter Tuning
Hyperparameter optimization was performed on the Random Forest model using **RandomizedSearchCV**.

Tuned parameters include:
- `n_estimators`
- `max_depth`
- `max_features`
- `min_samples_split`

This helped improve model generalization and reduce overfitting.

---

## Model Saving
The final tuned Random Forest model was saved using **pickle** for future use:

```python
with open('flight_price_rf_final_model.pkl', 'wb') as file:
    pickle.dump(best_rf, file)
```

## Future Improvements
Save and deploy a complete ML pipeline  
Experiment with advanced models like XGBoost or LightGBM  
