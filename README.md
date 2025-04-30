#  Health Risk Prediction with Explainable ML (SHAP)

This project builds a machine learning model to predict heart disease risk from patient clinical data. It also uses **SHAP (SHapley Additive Explanations)** to explain model predictions, promoting interpretability and trust in healthcare AI.

---

##  Project Structure

HealthRiskPrediction/ ├── heart.csv # Dataset 
├── step1_explore_data.py # Basic data overview 
├── step3_features_target.py # Feature/target separation 
├── step4_train_model.py # Initial model training 
├── step5_shap_explainer.py # SHAP + model tuning + metrics 


---

## What the Project Does

- Trains a **Random Forest Classifier** on clinical heart disease data
- Applies **hyperparameter tuning** with `GridSearchCV` (accuracy ~87%)
- Evaluates using confusion matrix, precision, recall, F1-score
- Uses **SHAP** to explain both global and individual predictions
- Visualizes SHAP impact using a **beeswarm plot**

---

## Features Used

- `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`
- `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

---

## Final Metrics (after tuning)

- **Accuracy**: 86.89%
- **Precision & Recall**: 0.86–0.88
- **Balanced performance** with trustworthy predictions

---

## SHAP Summary Plot

> SHAP beeswarm plot shows which features impact the model's predictions and by how much.
*Red = High feature value
Blue = Low feature value
*Purple/Pink = Medium or in-between values
->The farther a dot is from 0 (left or right), the more impact that feature had on the prediction.
->Dots to the right push the model toward predicting "disease (1)", and to the left toward "no disease (0)".
  
---

## Technologies Used

- Python 3.10
- Pandas
- Scikit-learn
- SHAP
- Matplotlib
- Seaborn (for confusion matrix heatmap)

---

## Author
Jyothi Surla

---

