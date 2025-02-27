#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importar librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LogisticRegression


# In[2]:


# Rutas de los archivos
contract_path = '/datasets/final_provider/contract.csv'
personal_path = '/datasets/final_provider/personal.csv'
internet_path = '/datasets/final_provider/internet.csv'
phone_path = '/datasets/final_provider/phone.csv'

# Cargar los datos
df_contract = pd.read_csv(contract_path)
df_personal = pd.read_csv(personal_path)
df_internet = pd.read_csv(internet_path)
df_phone = pd.read_csv(phone_path)


# In[3]:


# Unificar los Datasets
merged_data = df_contract.merge(df_personal, on="customerID", how = "left")
merged_data = merged_data.merge(df_internet, on= "customerID", how = "left")
merged_data = merged_data.merge(df_phone, on = "customerID", how = "left")


# In[4]:


# Convertir TotalCharges a numérico
merged_data['TotalCharges'] = pd.to_numeric(merged_data['TotalCharges'], errors='coerce')
missing_total_charges = merged_data['TotalCharges'].isnull().sum()
print(f"Valores faltantes en TotalCharges: {missing_total_charges}")

# Análisis de valores faltantes
nomissing_row = merged_data[merged_data["TotalCharges"].notnull()].head()
missing_rows = merged_data[merged_data['TotalCharges'].isnull()]

print("\nEjemplo de filas SIN valores faltantes en TotalCharges")
print(nomissing_row[['customerID', 'TotalCharges', 'MonthlyCharges', 'Type']])

print("\nFilas con valores faltantes en TotalCharges:")
print(missing_rows[['customerID', 'TotalCharges', 'MonthlyCharges', 'Type']])


# In[5]:


# Visualizar distribución de MonthlyCharges para filas con valores faltantes
plt.figure(figsize=(8, 5))
sns.histplot(missing_rows['MonthlyCharges'], kde=True, bins=20)
plt.title('Distribución de MonthlyCharges en valores faltantes de TotalCharges')
plt.xlabel('MonthlyCharges')
plt.ylabel('Frecuencia')
plt.show()

# Análisis de valores faltantes por tipo de contrato
missing_by_type = missing_rows['Type'].value_counts()
print("\nDistribución de tipos de contrato en valores faltantes:")
print(missing_by_type)


# In[6]:


# Imputación de valores faltantes usando la mediana por tipo de contrato
merged_data['TotalCharges'] = merged_data.groupby('Type')['TotalCharges'].transform(
    lambda x: x.fillna(x.median())
)
#Esta desición se tomo debido a que el numero de valores faltantes es pequeño por lo que no se considero necesario utilizar un modelo mas complejo
# Verificación de los valores faltantes despúes de imputar con la media
remaining_missing = merged_data['TotalCharges'].isnull().sum()
print(f"Valores faltantes restantes en TotalCharges después de imputación: {remaining_missing}")


# In[7]:


print(df_contract.info())


# # Creación del modelo de predicción

# In[8]:


# Definición de la columna objetivo
merged_data["Churn"] = merged_data["EndDate"].apply(lambda x: 0 if x == "No" else 1)

# Indica si el cliente tiene servicio de internet o no, ya que esto no está explícito en los datos
merged_data["has_internet"] = merged_data["InternetService"].notnull().astype(int)

# Indica si el cliente tiene servicio de telefonía o no, ya que esto no está explícito en los datos
merged_data["has_phone"] = merged_data["MultipleLines"].notnull().astype(int)

# División de los datos de entrenamiento
feature_columns = [
    'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
    'has_internet', 'has_phone', 'Type_One year', 'Type_Two year',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check', 'StreamingTV_Yes', 'OnlineSecurity_Yes'
]

# Convertir variables categóricas a variables dummy
merged_data = pd.get_dummies(
    merged_data, 
    columns=['Type', 'PaymentMethod', 'StreamingTV', 'OnlineSecurity'], 
    drop_first=True
)

# lista de columnas creadas tras la conversión a variables dummy
generated_columns = [
    col for col in merged_data.columns 
    if any(feature in col for feature in ['Type_', 'PaymentMethod_', 'StreamingTV_', 'OnlineSecurity_'])
]

# Actualizar la lista de columnas de características
feature_columns = ['MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'has_internet', 'has_phone'] + generated_columns

# Clasificación o selección de características y objetivo
features = merged_data[feature_columns]
target = merged_data['Churn']


# In[9]:


# Datos en entrenamiento y prueba
features_train, features_test, target_train, target_test = train_test_split(features,target,test_size=0.2, random_state= 12345)

# Entrenamiento del modelo
model = RandomForestClassifier(
    n_estimators=700, max_depth=10, min_samples_split=2, 
    min_samples_leaf=5, random_state=12345, class_weight="balanced"
)
model.fit(features_train, target_train)

# Evaluación del Modelo
target_pred = model.predict(features_test)
target_proba = model.predict_proba(features_test)[:, 1]

# Calcular AUC-ROC
auc_score = roc_auc_score(target_test, target_proba)
print(f"AUC-ROC: {auc_score:.2f}")

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(target_test, target_pred))

# Visualización de Importancia de Características
feature_importances = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
feature_importances.plot(kind='bar', title='Importancia de las Características')
plt.show()


# In[10]:


# Características más relevantes eliminando las que se consideren menos utilez para el entrenamiento del modelo
feature_columns = ['TotalCharges', 'MonthlyCharges', 'Type_Two year', 
                   'PaymentMethod_Electronic check', 'Type_One year', 
                   'has_internet', 'OnlineSecurity_Yes']

features = merged_data[feature_columns]
target = merged_data['Churn']

# Datos de entrenamiento y prueba
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=12345
)

# Entrenar el modelo nuevamente
model = RandomForestClassifier(n_estimators=700, max_depth=10, 
                               min_samples_split=2, min_samples_leaf=5, 
                               random_state=12345, class_weight="balanced")
model.fit(features_train, target_train)

# Evaluar el modelo después de eliminar variables irrelevantes
target_pred = model.predict(features_test)
target_proba = model.predict_proba(features_test)[:, 1]
auc_score = roc_auc_score(target_test, target_proba)

print(f"AUC-ROC después de eliminar variables irrelevantes: {auc_score:.2f}")


# In[11]:


from sklearn.model_selection import train_test_split, GridSearchCV

param_grid = {
    'n_estimators': [200, 500, 700],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}

rf = RandomForestClassifier(random_state=12345, class_weight='balanced')

grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', verbose=2, n_jobs=-1)
grid_search.fit(features_train, target_train)

# Mejor modelo encontrado
best_rf = grid_search.best_estimator_
print(f"Mejores parámetros: {grid_search.best_params_}")


# In[17]:


import shap
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Generación de nuevas características
merged_data['DuracionContrato'] = (pd.to_datetime(merged_data['EndDate'], errors='coerce') - pd.to_datetime(merged_data['BeginDate'])).dt.days.fillna(0)
merged_data['ChargeRatio'] = merged_data['TotalCharges'] / merged_data['MonthlyCharges']
merged_data['AvgMonthlyCharges'] = merged_data['TotalCharges'] / (merged_data['DuracionContrato'] + 1)

# Dividir datos en entrenamiento, validación y prueba
features = merged_data[['MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'has_internet', 'ChargeRatio', 'AvgMonthlyCharges']]
target = merged_data['Churn']

# Primero dividimos en entrenamiento (70%) y el resto (30% para validación y prueba)
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=12345)

# Luego dividimos el 30% restante en validación (50% de los datos restantes) y prueba (50%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=12345)

# Entrenar el modelo de CatBoost
cat_model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.01, loss_function='Logloss', random_seed=12345, verbose=100)
cat_model.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de validación
y_val_pred_proba = cat_model.predict_proba(X_val)[:, 1]
auc_score_val = roc_auc_score(y_val, y_val_pred_proba)
print(f"AUC-ROC en el conjunto de validación (CatBoost Mejorado): {auc_score_val:.2f}")

# Evaluar el modelo en el conjunto de prueba
y_test_pred_proba = cat_model.predict_proba(X_test)[:, 1]
auc_score_test = roc_auc_score(y_test, y_test_pred_proba)
print(f"AUC-ROC en el conjunto de prueba: {auc_score_test:.2f}")

# Interpretación con SHAP
explainer = shap.TreeExplainer(cat_model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)


# In[13]:


from sklearn.model_selection import KFold, cross_val_score

# Realizar validación cruzada K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=12345)

# Evaluar CatBoost usando AUC-ROC en cada partición
cv_scores = cross_val_score(cat_model, features, target, cv=kf, scoring='roc_auc')
print(f"AUC-ROC con K-Fold: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")


# In[14]:


print(X_train.shape, X_val.shape, X_test.shape)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# predicciones de clases
y_valid_pred_class = cat_model.predict(X_val)

#matriz de confusión
cm = confusion_matrix(y_val, y_valid_pred_class)

# Visualización
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cat_model.classes_)
disp.plot(cmap='Blues')


# In[ ]:




