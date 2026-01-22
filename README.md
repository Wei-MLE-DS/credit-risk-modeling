# Credit Risk Modeling Project Documentation

## 1. Data Analysis & Distributions

### Target Variable Distribution

The dataset is highly imbalanced, which is typical for credit risk data.

- **Target Feature:** `default_12m` (1 = Default, 0 = Non-Default)
- **Default Rate:** **2.85%**
- **Class Counts:**
  - Non-Default (0): 48,574
  - Default (1): 1,426

### Feature Distributions

Key financial features were analyzed to understand borrower profiles:

- **FICO Score:** A significant predictor; distributions show a clear separation where lower scores correlate with higher default rates. The analysis identified 2,500 (5.0%) missing values (encoded as 99999).
- **Loan Amount:** Ranges from $1,000 to $50,000, with a distribution skewed towards lower amounts.
- **Debt-to-Income (DTI):** Ranged up to ~0.97, indicating some borrowers have very high debt burdens.
- **Inquiries (Last 6 Months):** A discrete count variable where higher numbers typically indicate higher risk. About 2% of values were missing (encoded as 99).

## 2. Preprocessing & Methodology

### Handling Missing Values

- **Status:** Missing values were identified in `fico_score` (5.0%), `income` (3.0%), and `inquiries_last_6m` (2.0%).
- **Reasoning:** Handling missing values was crucial because standard linear models (like Logistic Regression) cannot process NaNs. Additionally, "missing" in credit data is often informative (e.g., a missing FICO score might imply a "thin file" or no credit history rather than random error).
- **Method:** The analysis checked if missingness itself correlated with default risk. For models requiring complete data, imputation strategies were applied.
- **Imputation:** Using median values from the training dataset to impute numerical features and using the most frequent values from the training dataset to impute categorical features.

### Handling Class Imbalance

- **Method Selected:** **`class_weight='balanced'`** (in Logistic Regression) and scale weights (in XGBoost).
- **Reasoning:** The project opted for cost-sensitive learning (adjusting weights) rather than resampling methods like SMOTE or undersampling.
  - *Efficiency:* Reweighting samples is computationally more efficient than generating synthetic data (SMOTE) for large datasets.
  - *Preservation:* It utilizes the full dataset without discarding majority class examples (undersampling) or introducing synthetic noise (oversampling).

### Data Splitting Strategy (Out-of-Time Validation)

Instead of a standard random train-test split, this project utilizes a **time-based split** using the `origination_date` column.

**Method:**

- **Training Set:** Loans originated before 2022-06-30
- **Validation Set:** Loans originated between 2022-07-01 and 2022-12-31
- **OOT (Out-of-Time) Set:** Loans originated after 2023-01-01

**Why this approach?**

- **Prevent Data Leakage:** Credit risk is highly dependent on macroeconomic cycles. Randomly splitting data would mix "future" knowledge (e.g., a recession or rate hike period) into the training set, causing the model to cheat.
- **Simulate Production:** In a real-world scenario, a model is trained on historical data to score new applications. Time-based splitting mirrors this reality.
- **Validate Robustness:** Testing on an "Out-of-Time" (OOT) dataset ensures the model remains stable and predictive even as borrower behaviors or economic conditions shift over time.

## 3. Model Comparison

Two primary models were trained and evaluated: **Logistic Regression** (baseline) and **XGBoost** (champion).

| Metric (Validation) | Logistic Regression | XGBoost |
| --- | --- | --- |
| **AUC-ROC** | 0.6730 | **0.7005** |
| **KS Statistic** | 0.2807 | **0.3186** |
| **F1-Score** | 0.1174 | 0.1015 |

- **Reason for Selection:** **XGBoost** was selected as the final model.
  - It demonstrated superior discrimination power (higher AUC and KS statistic).
  - It effectively captured non-linear relationships and interactions between features (e.g., Income vs. DTI) that the linear Logistic Regression model missed.

## 4. Feature Importance

Using SHAP (SHapley Additive exPlanations) values, the key drivers of default risk were identified:

1. **FICO Score:** The strongest predictor. Lower scores significantly increase default probability.
2. **Debt-to-Income (DTI) Ratio:** The second strongest driver. Higher ratios indicate higher risk.
3. **Income:** Lower income contributes to higher risk, though less dominantly than FICO or DTI.
4. **Utilization Rate:** Higher credit utilization correlates with higher risk.
5. **Secondary Factors:** Features like `state`, `channel`, and `product_type` had minimal impact compared to financial health indicators.

## 5. Production Pipeline & Artifacts

### Modular Function Script

All preprocessing and modeling functions have been encapsulated in a **Python script (`credit_risk_feature_engineering.py`)**. This allows direct function calls without running the notebook.

### Pipeline Setup

The final pipeline integrates:

- **Feature Validation:** Ensures the input dataframe matches expected feature order.
- **Preprocessing Steps:** Encapsulated in a `Pipeline` object.
- **Model Prediction:** XGBoost model with optimized threshold.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib
import pandas as pd

# Load artifacts
artifacts = joblib.load("credit_default_model.joblib")
pipeline = artifacts["pipeline"]
threshold = artifacts["threshold"]

# Predict function
def predict(df: pd.DataFrame):
    proba = pipeline.predict_proba(df)[:, 1]
    decision = (proba >= threshold).astype(int)
    return {
        "probability": proba.tolist(),
        "decision": decision.tolist()
    }
