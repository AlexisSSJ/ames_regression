from mlxtend.feature_selection import ColumnSelector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from plydata.one_table_verbs import pull
from mizani.formatters import comma_format, dollar_format
from plotnine import *
from siuba import *

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor

def adjusted_r2_score(y_true, y_pred, n, p):
  r2 = r2_score(y_true, y_pred)
  adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
  return adjusted_r2


transformed_df =get_df.transform(ames_x_train)
transformed_test = get_df.transform(ames_x_test)
"""
C R O S S   V A L I D A T I O N

A continuación se hace dos procesos de CV, para decidir si predecir
el precio total de la casa o el precio por metro cuadrado.

Se iteran con los
  weights: ['uniform', 'distance']
  metric: ['euclidean', 'manhattan','haversine', 'cosine', 'correlation']
  p: [5,15,20]

"""
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

param_grid = {
 'n_neighbors': range(3, 60 ,2),
 'weights': ['uniform', 'distance'],
 'metric': ['euclidean', 'manhattan','haversine', 'cosine', 'correlation', 'mahalanobis','minkowski'],
 'p' : [5,15,20]
}

"""
Cross validation para precio por SF
"""

scoring = {
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    'r2': make_scorer(adjusted_r2_score, greater_is_better=True, n=np.ceil(len(ames_x_train)), p=transformed_df.shape[1]),
    'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
    'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
}


pipeline_knn_cv = Pipeline([
  ('clean_data', clean_and_imputers),
  ('feat_eng', feature_eng),
  ('preprocessor', preprocessor_1),
  ('interactions', interactions),
  ('regressor', GridSearchCV(
      KNeighborsRegressor(),
      param_grid,
      cv=kf,
      scoring=scoring,
      refit='neg_mean_squared_error',
      verbose=3,
      n_jobs=7)
     )]
)
pd.set_option('display.max_columns', 500)
pipeline_knn_cv.fit(ames_x_train, ames_y_train)

results_cv_trans = pipeline_knn_cv.named_steps['regressor']
# pd.to_pickle(results_cv_trans, 'model/knn/cv_trans_sp.pkl')

results_df_trans = pd.DataFrame(results_cv_trans.cv_results_)

summary_cv_t = (results_df_trans >>select(-_.contains("split"), -_.contains("time"), -_.params))

check_cv(summary_cv_t, param_p=False)
  
"""
Cross validation para precio total de la casa
"""
pipeline_knn_cv.fit(ames_x_train, Sale_Price_train)

results_cv_ = pipeline_knn_cv.named_steps['regressor']
pd.to_pickle(results_cv_, 'model/knn/cv_sp.pkl')
results_df_ = pd.DataFrame(results_cv_.cv_results_)

summary_cv_ = (results_df_ >>select(-_.contains("split"), -_.contains("time"), -_.params))

check_cv(summary_cv_)
knn_reg = results_cv_.best_estimator_

# Los resultados muestran que KNN tiene un mejor performance en promedio
# prediciendo el precio total de la casa

""" 
Selection de variables:
  Para la seleccion de variables se hara un bucle para
  poder identificar variables poco importantes e irlas 
  removiendo del modelo identificando con que variables 
  se tiene el mejor performance a traves de la R^2 y MSE
"""


predictores = transformed_test.columns.shape[0]
best_mods = []
score_mse =[]
score_r = []

for k in range (0,185,5):
  t= 207-k
  if k == 0:
    sc = transformed_df.columns.to_list()
  else: 
    sc = (results >> filter (_.P==t) >> pull ('used_vars'))[0]

  knn_reg.fit(transformed_df[sc], Sale_Price_train)
  metrics = get_metrics(knn_reg.predict(transformed_test[sc]), Sale_Price_test, len(sc))
  importance_df = importance_from_model(
    test_frame = transformed_test,
    y_obs = Sale_Price_test ,
    selected_columns =  sc, 
    pipeline = knn_reg,
    actual_mse = metrics.loc['MSE'].values[0] ,
    n_permutations= 50,
    trans_pred= False
  )
  
  vars_importance_order = list( importance_df >> arrange(_.Mean_Loss) >> pull('Variable'))
  transformed_df =get_df.transform(ames_x_train)
  
  v=[]
  r2=[]
  mse_=[]
  p=[]
  for i in range(t):
    vars_to_train = vars_importance_order[i:]
    p.append(len(vars_to_train))
    knn_reg.fit(transformed_df[vars_to_train], Sale_Price_train)
    pred= knn_reg.predict(transformed_test[vars_to_train])
    mse_.append(get_metrics(pred,Sale_Price_test, len(vars_to_train)).loc['MSE'].values[0])
    r2.append(get_metrics(pred,Sale_Price_test, len(vars_to_train)).loc['R2Adj'].values[0])
    v.append(vars_to_train)
    
  results = pd.DataFrame() >> mutate(MSE = mse_, P=p,R2=r2,used_vars = v)
  best_mods.append( (results >> filter (_.R2>0.7) >> pull ('used_vars')) )
  score_r.append( (results >> filter (_.R2>0.7) >> pull ('R2')) )
  score_mse.append( (results >> filter (_.R2>0.7) >> pull ('MSE')) )


scoresr =  [item for sub_list in score_r for item in sub_list]
scoresmse=  [item for sub_list in score_mse for item in sub_list]
best_mods_ = [item for sub_list in best_mods for item in sub_list] #best_models_df.used_vars.astype(str).str.split('(?<=\'\]), (?=\[)', regex = True).explode()
num_pred = [len(l) for l in best_mods_]


results_knn_model_iterations = pd.DataFrame() >> mutate (used_vars = best_mods_, r2 = scoresr ,mse = scoresmse, p= num_pred) 

results_knn_model_iterations.to_csv ('model/knn/interations_results_knn.csv')
(
  results_knn_model_iterations
    >> ggplot (aes(x='p', y= 'r2' ,color= 'mse'))
    +  geom_point(alpha = 0.6)
    +  labs (x= 'Número de predictores', y= 'R^2', color = 'MSE', title = 'Resultados de iteraciónes con dif. combinaciones de predictores')
)

(
  results_knn_model_iterations >> filter (_.r2>0.85)
    >> ggplot (aes(x='p', y= 'r2' ,color= 'mse'))
    +  geom_point(alpha = 0.6)
    +  labs (x= 'Número de predictores', y= 'R^2', color = 'MSE', title = 'Resultados de iteraciónes con dif. combinaciones de predictores')
)


# results_knn_model_iterations.to_csv('data/knn_mod_it.csv')
results_knn_model_iterations >> top_n(50, _.r2) >> arrange (_.p) >> select(-_.used_vars)



selected_vars = results_knn_model_iterations.loc[3450].used_vars
""" 
Se encotró un modelo con con buenas características. 
Ahora se hará un CV para buscar el mejor hiperparametro K
y distancia
"""
step_select =ColumnTransformer(
  [('selector', 'passthrough', selected_vars )],
  verbose_feature_names_out = False,
  remainder='drop')


knn_regressor = Pipeline([
  ('feat_eng', get_df),
  ('feature_selection', step_select),
  ('regressor', KNeighborsRegressor(
    n_neighbors = 4 ,
    metric = 'manhattan',
    weights = 'distance'))
    ])

pipeline_knn_regressor.fit(ames_x_train, Sale_Price_train)

pd.to_pickle(knn_regressor, 'model/knn/best_knn_model.pkl')

y_pred = pipeline_knn_regressor.predict(ames_x_test)
y_obs = Sale_Price_test

get_metrics(y_pred, y_obs, len(selected_vars))

get_metrics( pipeline_knn_regressor.predict(ames_x_val), Sale_Price_validation, len(selected_vars))
