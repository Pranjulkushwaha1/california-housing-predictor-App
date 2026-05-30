# California Housing Price Predictor

## About
A machine learning web app that predicts California house prices based on location and housing features.

## Technologies Used
- Python
- XGBoost
- Scikit-learn
- Streamlit
- Pandas, NumPy

## Steps Followed
- EDA (Exploratory Data Analysis)
- Missing Value Handling
- Feature Engineering
- One Hot Encoding
- Train Test Split
- StandardScaler
- Linear Regression, Random Forest, XGBoost

## Results
| Model | R2 Score |
|---|---|
| Linear Regression | 0.597 |
| Random Forest | 0.805 |
| XGBoost | 0.824 |

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```