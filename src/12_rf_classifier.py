from mlxtend.feature_selection import ColumnSelector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from plydata.one_table_verbs import pull
from mizani.formatters import comma_format, dollar_format
from plotnine import *
from siuba import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle


#### CARGA DE DATOS ####
telco = pd.read_csv("data/Churn.csv")

telco_y = telco >> pull("Churn")    # telco[["Churn"]]
telco_x =  select(telco, -_.Churn, -_.customerID)   # telco.drop('Churn', axis=1)

#### DIVISIÓN DE DATOS ####
telco_x_train, telco_x_test, telco_y_train, telco_y_test = train_test_split(
 telco_x, telco_y, 
 train_size = 0.80, 
 random_state = 195,
 stratify = telco_y
 )


#### FEATURE ENGINEERING ####

## SELECCIÓN DE VARIABLES

# Seleccionamos las variales numéricas de interés
num_cols = ["MonthlyCharges"]

# Seleccionamos las variables categóricas de interés
cat_cols = ["PaymentMethod", "Dependents"]

# Juntamos todas las variables de interés
columnas_seleccionadas = num_cols + cat_cols

pipe = ColumnSelector(columnas_seleccionadas)
telco_x_train_selected = pipe.fit_transform(telco_x_train)

telco_train_selected = pd.DataFrame(
  telco_x_train_selected, 
  columns = columnas_seleccionadas
  )

telco_train_selected.info()


## TRANSFORMACIÓN DE COLUMNAS

# ColumnTransformer para aplicar transformaciones
preprocessor = ColumnTransformer(
    transformers = [
        ('scaler', StandardScaler(), num_cols),
        ('onehotencoding', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough'  # Mantener las columnas restantes sin cambios
)

transformed_data = preprocessor.fit_transform(telco_train_selected)
new_column_names = preprocessor.get_feature_names_out()

transformed_df = pd.DataFrame(
  transformed_data,
  columns=new_column_names
  )

transformed_df
transformed_df.info()


#### PIPELINE Y MODELADO

# Crear el pipeline con la regresión lineal
pipeline = Pipeline([
   ('preprocessor', preprocessor),
   ('regressor', RandomForestClassifier(n_estimators=10,
     min_samples_split=2,
     min_samples_leaf=2,
     random_state=12345))
])

# Entrenar el pipeline
results = pipeline.fit(telco_train_selected, telco_y_train)

## PREDICCIONES
y_pred = pipeline.predict(telco_x_test)

telco_test = (
  telco_x_test >>
  mutate(Churn_Pred = y_pred, Churn = telco_y_test)
)

(
telco_test >>
  select(_.Churn, _.Churn_Pred)
)


# ----------------------------------------------#
#            METRICAS DE DESEMPEÑO 
# ----------------------------------------------#

matriz_confusion = confusion_matrix(telco_y_test, y_pred)
matriz_confusion


warnings.filterwarnings("ignore")

# Crear un DataFrame a partir de la matriz de confusión
confusion_df = pd.DataFrame(
  matriz_confusion, 
  columns=['Predicción Negativa', 'Predicción Positiva'], 
  index=['Real Negativa', 'Real Positiva']
  )

# Crear una figura utilizando Seaborn
plt.plot();
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', cbar=False);

plt.title('Matriz de Confusión');
plt.xlabel('Predicción');
plt.ylabel('Realidad');
plt.show();

# ----------------------------------------------#
#         ESTIMACION DE PROBABILIDADES 
# ----------------------------------------------#


y_pred = pipeline.predict_proba(telco_x_test)[:,0]
Churn_Pred = np.where(y_pred >= 0.7, "No", "Yes")

results = (
  telco_x_test >>
  mutate(
    Churn_Prob = y_pred, 
    Churn_Pred = Churn_Pred,
    Churn = telco_y_test) >>
  select(_.Churn_Prob, _.Churn_Pred, _.Churn)
)

results

(
  results
  >> group_by(_.Churn_Pred)
  >> summarize(n = _.Churn_Pred.count() )
)  


confusion_df = pd.DataFrame(
  confusion_matrix(telco_y_test, Churn_Pred), 
  columns=['Predicción Negativa', 'Predicción Positiva'], 
  index=['Real Negativa', 'Real Positiva']
  )

# Crear una figura utilizando Seaborn
plt.plot();
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', cbar=False);

plt.title('Matriz de Confusión');
plt.xlabel('Predicción');
plt.ylabel('Realidad');
plt.show();


fpr, tpr, thresholds = roc_curve(
  y_true = np.where(telco_y_test == "Yes", 0, 1), 
  y_score = y_pred
  )

roc_thresholds = pd.DataFrame({
  'thresholds': thresholds, 
  'tpr': tpr, 
  'fpr': fpr}
  )

(
  roc_thresholds >>
  ggplot(aes(x = fpr, y = tpr)) +
  geom_path(size = 1.2) +
  geom_abline(colour = "gray") +
  xlab("Tasa de falsos positivos") +
  ylab("Sensibilidad") +
  ggtitle("Curva ROC")
)

roc_auc_score(np.where(telco_y_test == "Yes", 0, 1), y_pred)


precision, recall, thresholds = precision_recall_curve(
  y_true = np.where(telco_y_test == "Yes", 0, 1),
  probas_pred = y_pred
  )
  
pr_thresholds = pd.DataFrame({
  'thresholds': np.append(0, thresholds), 
  'precision': precision, 
  'recall': recall}
  )

(
  pr_thresholds >>
  ggplot(aes(x = recall, y =precision)) +
  geom_path(size = 1.2) +
  geom_abline(colour = "gray", intercept = 1, slope = -1) +
  xlim(0, 1) + ylim(0, 1) +
  xlab("Recall") +
  ylab("Precision") +
  ggtitle("Curva PR")
)

average_precision_score(np.where(telco_y_test == "Yes", 0, 1), y_pred)


# ----------------------------------------------#
#         VALIDACION CRUZADA 
# ----------------------------------------------#

# Definir el objeto K-Fold Cross Validator
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

param_grid = {
 'max_depth': range(2, 5),
 'min_samples_split': range(2, 8),
 'min_samples_leaf': range(2, 8),
 'max_features': range(1, 5)
}

# Algunas otras posibles distancias son:
# ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming', 'jaccard', 'cosine']

# Definir las métricas de desempeño que deseas calcular como funciones de puntuación


scoring = {
  'roc_auc': make_scorer(roc_auc_score, greater_is_better=True),
  'average_precision': make_scorer(average_precision_score, greater_is_better=True)
  }

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GridSearchCV(
      RandomForestClassifier(), 
      param_grid, 
      cv=kf, 
      scoring=scoring, 
      refit='average_precision',
      verbose=3, 
      n_jobs=7)
     )
])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(telco_y_train)

pipeline.fit(telco_train_selected, 1-y)
pickle.dump(pipeline, open('models/grid_search_random_forest_class.pkl', 'wb'))
pipeline = pickle.load(open('models/grid_search_random_forest_class.pkl', 'rb'))

results_cv = pipeline.named_steps['regressor'].cv_results_

# Convierte los resultados en un DataFrame
pd.set_option('display.max_columns', 500)
results_df = pd.DataFrame(results_cv)
results_df.columns

# Puedes seleccionar las columnas de interés, por ejemplo:

summary_df = (
  results_df >>
  select(-_.contains("split._"), -_.contains("time"), -_.params)
)
summary_df

(
  summary_df >>
  ggplot(aes(x = "param_max_features", y = "mean_test_roc_auc")) +
  geom_point() +
  ggtitle("Parametrización de Random Forest vs ROC AUC") +
  xlab("Parámetro: Número de vecinos cercanos") +
  ylab("ROC AUC promedio")
)

(
  summary_df >>
  select(_.param_max_depth, _.param_max_features, _.param_min_samples_leaf, 
         _.param_min_samples_split, _.mean_test_roc_auc) >>
  pivot_longer(
    cols = ["param_max_depth", "param_max_features", "param_min_samples_leaf", "param_min_samples_split"],
    names_to="parameter",
    values_to="value") >>
  ggplot(aes(x = "value", y = "mean_test_roc_auc")) +
  geom_point(size = 1, ) +
  facet_wrap("~parameter", scales = "free_x") +
  xlab("Parameter value") +
  ylab("R^2 promedio") +
  ggtitle("Parametrización de Random Forest vs ROC AUC")
)


best_params = pipeline.named_steps['regressor'].best_params_
best_params
best_estimator = pipeline.named_steps['regressor'].best_estimator_
best_estimator


## PREDICCIONES FINALES

final_rf_pipeline = Pipeline([
   ('preprocessor', preprocessor),
   ('regressor', best_estimator)
])

# Entrenar el pipeline
final_rf_pipeline.fit(telco_train_selected, telco_y_train)

## Predicciones finales
y_pred_rf = final_rf_pipeline.predict_proba(telco_x_test)[:,0]
Churn_Pred = np.where(y_pred_rf >= 0.7, "No", "Yes")

results = (
  telco_x_test >>
  mutate(
    Churn_Prob = y_pred_rf, 
    Churn_Pred = Churn_Pred,
    Churn = telco_y_test) >>
  select(_.Churn_Prob, _.Churn_Pred, _.Churn)
)

results

(
  results
  >> group_by(_.Churn_Pred)
  >> summarize(n = _.Churn_Pred.count() )
)  

confusion_df = pd.DataFrame(
  confusion_matrix(telco_y_test, Churn_Pred), 
  columns=['Predicción Negativa', 'Predicción Positiva'], 
  index=['Real Negativa', 'Real Positiva']
  )


# Crear una figura utilizando Seaborn
plt.plot();
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', cbar=False);

plt.title('Matriz de Confusión');
plt.xlabel('Predicción');
plt.ylabel('Realidad');
plt.show();


fpr, tpr, thresholds = roc_curve(
  y_true = np.where(telco_y_test == "Yes", 0, 1), 
  y_score = y_pred_rf
  )

roc_thresholds = pd.DataFrame({
  'thresholds': thresholds, 
  'tpr': tpr, 
  'fpr': fpr}
  )

(
  roc_thresholds >>
  ggplot(aes(x = fpr, y = tpr)) +
  geom_path(size = 1.2) +
  geom_abline(colour = "gray") +
  xlab("Tasa de falsos positivos") +
  ylab("Sensibilidad") +
  ggtitle("Curva ROC")
)

roc_auc_score(np.where(telco_y_test == "Yes", 0, 1), y_pred)


precision, recall, thresholds = precision_recall_curve(
  y_true = np.where(telco_y_test == "Yes", 0, 1),
  probas_pred = y_pred_rf
  )
  
pr_thresholds = pd.DataFrame({
  'thresholds': np.append(0, thresholds), 
  'precision': precision, 
  'recall': recall}
  )

(
  pr_thresholds >>
  ggplot(aes(x = recall, y =precision)) +
  geom_path(size = 1.2) +
  geom_abline(colour = "gray", intercept = 1, slope = -1) +
  xlim(0, 1) + ylim(0, 1) +
  xlab("Recall") +
  ylab("Precision") +
  ggtitle("Curva PR")
)


roc_auc = average_precision_score(np.where(telco_y_test == "Yes", 0, 1), y_pred_rf)


#### Importancia de variables


# 1 permutación

importance = np.zeros(telco_x_test[columnas_seleccionadas].shape[1])

# Realiza el procedimiento de permutación
for i in range(telco_x_test[columnas_seleccionadas].shape[1]):
    telco_x_test_permuted = telco_x_test[columnas_seleccionadas].copy()
    telco_x_test_permuted.iloc[:, i] = shuffle(telco_x_test_permuted.iloc[:, i], random_state=42)  
    # Permuta una característica
    y_pred_permuted = final_rf_pipeline.predict_proba(telco_x_test_permuted)[:,0]
    
    roc_auc_permuted = roc_auc_score(np.where(telco_y_test == "Yes", 0, 1), y_pred_permuted)
    
    importance[i] = roc_auc - roc_auc_permuted


# Calcula la importancia relativa
importance = importance / importance.sum()
importance

importance_df = pd.DataFrame({
  'Variable': columnas_seleccionadas, 
  'Importance': importance
  })

# Crea la gráfica de barras
(
  importance_df >>
  ggplot(aes(x= 'reorder(Variable, Importance)', y='Importance')) + 
  geom_bar(stat='identity', fill='blue', color = "black") + 
  labs(title='Importancia de las Variables', x='Variable', y='Importancia') +
  coord_flip() +
  ylim(0, 0.5)
)


# N permutaciones

n_permutations = 50
performance_losses = []

for i in range(telco_x_test[columnas_seleccionadas].shape[1]):
    loss = []
    for j in range(n_permutations):
        telco_x_test_permuted = telco_x_test[columnas_seleccionadas].copy()
        telco_x_test_permuted.iloc[:, i] = np.random.permutation(telco_x_test_permuted.iloc[:, i])
        y_pred_permuted = final_rf_pipeline.predict_proba(telco_x_test_permuted)[:,0]
        # Permuta una característica
        roc_auc_permuted = roc_auc_score(np.where(telco_y_test == "Yes", 0, 1), y_pred_permuted)
        
        loss.append(roc_auc_permuted)
    performance_losses.append(loss)

performance_losses = performance_losses/np.sum(performance_losses, axis=0)
mean_losses = np.mean(performance_losses, axis=1)
std_losses = np.std(performance_losses, axis=1)

importance_df = pd.DataFrame({
  'Variable': columnas_seleccionadas, 
  'Mean_Loss': mean_losses, 
  'Std_Loss': std_losses
  })

(
  importance_df >>
  mutate(
    ymin = _.Mean_Loss - _.Std_Loss,
    ymax = _.Mean_Loss + _.Std_Loss) >>
  ggplot(aes(x = 'reorder(Variable, Mean_Loss)', y = "Mean_Loss")) +
  geom_errorbar(aes(ymin='ymin', ymax='ymax'),
    width=0.1, position=position_dodge(0.9)) +
  geom_point(alpha = 0.65) +
  labs(title='Importancia de las Variables', x='Variable', y='Importancia') +
  coord_flip() +
  ylim(0, 0.5)
)

importances = final_rf_pipeline.named_steps['regressor'].feature_importances_
columns = final_rf_pipeline["preprocessor"].get_feature_names_out()

(
  pd.DataFrame({'feature': columns, 'Importance': importances}) >>
  ggplot(aes(x= 'reorder(feature, Importance)', y='Importance')) + 
  geom_bar(stat='identity', fill='blue', color = "black") + 
  labs(title='Importancia de las Variables', x='Variable', y='Importancia') +
  coord_flip()
)








