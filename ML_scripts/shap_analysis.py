import shap
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("ML_scripts/data.csv")
X = df.drop(columns=["formation_energy"])
y = df["formation_energy"])

model = GradientBoostingRegressor().fit(X, y)
explainer = shap.Explainer(model)
shap_values = explainer(X)
shap.summary_plot(shap_values, X)
