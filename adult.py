# %%
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from category_encoders.target_encoder import TargetEncoder
from sklearn.metrics import roc_auc_score
import shap

# %%
df = pd.read_csv("data/adult.csv")
df["class"] = np.where(df["class"] == "<=50K", 1, 0)
df["sex"] = np.where(df["sex"] == "Male", 1, 0)
# %%
protected = "sex"
## Split by half
df_tr = df.loc[:16280]
df_te = df.loc[16280:32560]
df_val = df.loc[32560:]

# %%
X_tr = df_tr.drop(columns="class").drop(columns=protected)
X_te = df_te.drop(columns="class").drop(columns=protected)
X_val = df_val.drop(columns="class").drop(columns=protected)
y_tr = df_tr[["class"]]
y_te = df_te[["class"]]
y_protected_test = df_val[protected]
y_protected_val = df_val[protected]

# %%
te = TargetEncoder()
X_tr = te.fit_transform(X_tr, y_tr)
X_te = te.transform(X_te)
X_val = te.transform(X_val)
# %%
model = XGBRegressor()
model.fit(X_tr, y_tr)
explainer = shap.Explainer(model)
shap_values = explainer(X_te)
shap_values = pd.DataFrame(shap_values.values,columns=X_tr.columns)

shap_values_val = explainer(X_val)
shap_values_val = pd.DataFrame(shap_values_val.values,columns=X_tr.columns)


# %%
clf = XGBClassifier()
clf.fit(shap_values_val,y_protected_test)
# %%
roc_auc_score(y_protected_val,clf.predict_proba(shap_values_val)[:,1],)
# %%
explainer = shap.Explainer(clf)
shap_values = explainer(X_val)
shap.plots.bar(shap_values)
