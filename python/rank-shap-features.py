import shap
import numpy as np
import pandas as pd

explainer = shap.Explainer(xgb_model)  
shap_values = explainer.shap_values(x_train)

feature_names = x_train.columns
shap_means = np.abs(shap_values).apply(np.mean, axis = 0)

feature_importance = pd.DataFrame({'feature': feature_names, 'mean_shap_value': shap_means}).sort_values(by='mean_shap_value', ascending=False).reset_index(drop=True)

feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
feature_importance.head()

