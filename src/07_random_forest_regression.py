from mlxtend.feature_selection import ColumnSelector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn import set_config

from plydata.one_table_verbs import pull
from plydata.tidy import pivot_longer
from mizani.formatters import comma_format, dollar_format
from plotnine import *
from siuba import *

import pandas as pd
import numpy as np
import pickle
pd.set_option('display.max_columns', 500)
pd.options.display.float_format = '{:.4f}'.format

"""
CROSS VALIDATION    

Se hace un CV inical para seleccionar parámetros para un mode
iniclal.
"""
# Definir el objeto K-Fold Cross Validator
k = 7
kf = KFold(n_splits=k, shuffle=True, random_state=42)

param_grid = {
 'max_depth': range(11, 18),
 'min_samples_split': range(2,9),
 'min_samples_leaf': range(2,6),
 'max_features': range(19,26,2)
}

transformed_df = get_df.fit_transform(ames_x_train)

scoring = {
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    'r2': make_scorer(adjusted_r2_score, greater_is_better=True, 
                      n=np.ceil(len(transformed_df)), p=len(transformed_df.columns)),
    'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
    'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
}



preliminar_cv = Pipeline([
    ('preprocessor', get_df),
    ('regressor', GridSearchCV(
      RandomForestRegressor(), 
      param_grid = param_grid, 
      cv=kf, 
      scoring=scoring, 
      refit='neg_mean_squared_error',
      verbose=2, 
      n_jobs=7,
      error_score='raise')
     )
])

preliminar_cv.fit(ames_x_train, ames_y_train)


"""
ANALISIS DEL PRIMEROS RESUTALSO DE CV-1
"""

# pickle.dump(pipeline.named_steps['regressor'], open('grid_search_random_forest_2.pkl', 'wb'))
results_cv = pickle.load(open('model/Random_forest/grid_search_random_forest_1.pkl', 'rb'))

# Convierte los resultados en un DataFrame

summary_df = pd.DataFrame(results_cv.cv_results_) >> select(-_.contains("split._"), -_.contains("time"), -_.params)



# Puedes seleccionar las columnas de interés, por ejemplo:
(
  summary_df >> top_n(20 ,  _.mean_test_r2 )>>
  select(_.param_max_depth, _.param_max_features, _.param_min_samples_leaf, 
         _.param_min_samples_split, _.mean_test_r2) >>
  pivot_longer(
    cols = ["param_max_depth", "param_max_features", "param_min_samples_leaf", "param_min_samples_split"],
    names_to="parameter",
    values_to="value") >>
  ggplot(aes(x = "value", y = "mean_test_r2")) +
  geom_point(size = 1, ) +
  facet_wrap("~parameter", scales = "free_x") +
  xlab("Parameter value") +
  ylab("R^2 promedio") +
  ggtitle("Parametrización de Random Forest vs R^2")
)


results_cv.best_params_


preliminar_regressor = Pipeline([
   ('preprocessor', get_df),
   ('regressor', results_cv.best_estimator_)
])

# Entrenar el pipeline
preliminar_regressor.fit(ames_x_train, ames_y_train)

"""
Análisis de importancia de las variables para el modelo preliminar
"""

importances  = pd.DataFrame() >> mutate (Variable = results_cv.best_estimator_.feature_names_in_, Importance = results_cv.best_estimator_.feature_importances_ )

y_pred = preliminar_regressor.predict(ames_x_test) * ames_x_test.Gr_Liv_Area
y_obs = Sale_Price_test

metrics = get_metrics(y_pred, y_obs, len(transformed_df.columns))
# La R ajustada salé negativa porque hay más predictores que datos en testing


## Graficas top 40 +-
(
importances 
    >> top_n(40, _.Importance)
    >>ggplot (aes(x = 'reorder(Variable,Importance)', y = 'Importance'))
    + geom_col()
    + labs(title = 'Top 40 variables con mayor importancia')
    + coord_flip()
)

(
importances 
    >> top_n(-40, _.Importance)
    >>ggplot (aes(x = 'reorder(Variable,Importance)', y = 'Importance'))
    + geom_col()
    + labs(title = 'Top 40 variables con menos importancia')
    + coord_flip()
)


"""
    SELECCIÓN DE VARIABLES ELMINANDO 1X1 POR IMPORTANCIA
"""


vars_importance_order = list( importances >> arrange(_.Importance) >> pull('Variable'))
transformed_test = get_df.transform(ames_x_test)


v=[]
r2=[]
mse_=[]
p=[]

best_estimator = RandomForestRegressor(
    n_estimators  = 500,
    max_depth = 14,
    max_features=23, 
    min_samples_leaf=2,
    random_state= 7,
    n_jobs = 7)
for i in range(50, 180):
    vars_to_train = vars_importance_order[i:]
    
    
    best_estimator.fit(transformed_df[vars_to_train], ames_y_train)
    pred= best_estimator.predict(transformed_test[vars_to_train]) * ames_x_test.Gr_Liv_Area
    
    
    p.append(len(vars_to_train))
    mse_.append(get_metrics(pred,Sale_Price_test, len(vars_to_train)).loc['MSE'].values[0])
    r2.append(get_metrics(pred,Sale_Price_test, len(vars_to_train)).loc['R2Adj'].values[0])
    v.append(vars_to_train)
    

# results_imp_tunning = pd.DataFrame() >> mutate (MSE = mse_, P= p, used_vars = v , R_adj = r2)
results_imp_tunning = pd.read_csv('model/Random_forest/importance_tunning_1.csv')
# results_imp_tunning.to_csv('model/Random_forest/importance_tunning_1.csv')
(
    results_imp_tunning 
        >> pivot_longer(cols=['MSE','R_adj'], names_to='metric', values_to='Score' )
        >> ggplot (aes (x = 'P',y= 'Score' ))
        +  geom_point()
        + facet_wrap("~metric",ncol = 1, scales  = 'free_y')
)

selected_vars = (results >> filter (_.P==34) >> pull("used_vars"))[0]

step_select =ColumnTransformer(
  [('selector', 'passthrough', selected_vars )],
  verbose_feature_names_out = False,
  remainder='drop').set_output(transform = "pandas")

"""
        Segunda Ronda de CV--------------------------------------------------
"""
param_grid = {
 'max_depth': range(11, 18),
 'min_samples_split': range(2,9),
 'min_samples_leaf': range(2,6),
 'max_features': range(19,26,2)
}


scoring = {
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    'r2': make_scorer(adjusted_r2_score, greater_is_better=True, 
                      n=np.ceil(len(transformed_df)), p=len(selected_vars)),
    'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
    'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
}

best_features_pipe= Pipeline([
    ('preprocessor', get_df),
    ('step_select', step_select),
    ('regressor', GridSearchCV(
      RandomForestRegressor(), 
      param_grid = param_grid, 
      cv=kf, 
      scoring=scoring, 
      refit='neg_mean_squared_error',
      verbose=2, 
      n_jobs=7,
      error_score='raise')
     )
])

best_features_pipe.fit(ames_x_train ,ames_y_train)
# results_cv_final = pipeline_cv.named_steps['regressor']


results_pipe = pickle.load(open('model/Random_forest/rf_1_cv.pkl', 'rb'))
# pickle.dump(pipeline_cv.named_steps['regressor'], open('model/Random_forest/rf_1_cv.pkl', 'wb'))

# Convierte los resultados en un DataFrame
summary_df = pd.DataFrame(results_pipe.cv_results_) >> select(-_.contains("split._"), -_.contains("time"), -_.params)

(
  summary_df >> top_n(30 ,  _.mean_test_r2 )>>
  select(_.param_max_depth, _.param_max_features, _.param_min_samples_leaf, 
         _.param_min_samples_split, _.mean_test_r2) >>
  pivot_longer(
    cols = ["param_max_depth", "param_max_features", "param_min_samples_leaf", "param_min_samples_split"],
    names_to="parameter",
    values_to="value") >>
  ggplot(aes(x = "value", y = "mean_test_r2")) +
  geom_point(size = 1, ) +
  facet_wrap("~parameter", scales = "free_x") +
  xlab("Parameter value") +
  ylab("R^2 promedio") +
  ggtitle("Parametrización de Random Forest vs R^2")
)

pipeline_cv.named_steps['regressor'].best_params_

rf2 = RandomForestRegressor(n_estimators = 70, max_depth=12, max_features=21, min_samples_leaf=4,
                      min_samples_split=5, random_state =7, n_jobs = 6)
                      
                      
pipeline_cv_final = Pipeline([
    ('preprocessor', get_df),
    ('step_select', step_select),
    ('regressor', rf2)
])


pipeline_cv_final.fit(ames_x_train, ames_y_train)


y_pred_rf = pipeline_cv_final.predict(ames_x_test)*ames_x_test.Gr_Liv_Area

results_reg = (
  ames_x_test >>
  mutate(final_rf_pred = y_pred_rf, 
        Sale_Price = Sale_Price_test) >>
  select(_.Sale_Price, _.final_rf_pred))

(
    results_reg
        >> ggplot (aes (x = 'final_rf_pred', y ='Sale_Price'))
        +  geom_point()
        + geom_abline(intercept = 0 ,slope =1)
)

y_obs = Sale_Price_test

get_metrics(y_pred_rf,y_obs, len(selected_vars))


pipeline_cv_final.named_steps['regressor'].feature_names_in_


importances  = pd.DataFrame() >> mutate (Variable =pipeline_cv_final.named_steps['regressor'].feature_names_in_, Importance = pipeline_cv_final.named_steps['regressor'].feature_importances_ )

y_pred = final_rf_pipeline.predict(ames_x_test) * ames_x_test.Gr_Liv_Area
y_obs = Sale_Price_test

metrics = get_metrics(y_pred, y_obs, len(transformed_df.columns))

(
importances >> 
                >> top_n(40, _.Importance)
                >>ggplot (aes(x = 'reorder(Variable,Importance)', y = 'Importance'))
                + geom_col()
                + labs(title = 'Top 40 variables con mayor importancia')
                + coord_flip()
)

(
pd.DataFrame() >> mutate (Variable = best_estimator.feature_names_in_,
                           Importance = best_estimator.feature_importances_ )
                >> top_n(-40, _.Importance)
                >>ggplot (aes(x = 'reorder(Variable,Importance)', y = 'Importance'))
                + geom_col()
                + labs(title = 'Top 40 variables con menos importancia')
                + coord_flip()
)



vars_importance_order = list( importances >> arrange(_.Importance) >> pull('Variable'))
transformed_test = get_df.transform(ames_x_test)


v=[]
r2=[]
mse_=[]
p=[]
best_estimator = RandomForestRegressor(
                                n_estimators = 70, 
                                max_depth=12, 
                                max_features=21, 
                                min_samples_leaf=4,
                                min_samples_split=5, 
                                random_state =7, 
                                n_jobs = 6)
for i in range( 20):
    vars_to_train = vars_importance_order[i:]
    
    
    best_estimator.fit(transformed_df[vars_to_train], ames_y_train)
    pred= best_estimator.predict(transformed_test[vars_to_train]) * ames_x_test.Gr_Liv_Area
    
    
    p.append(len(vars_to_train))
    mse_.append(get_metrics(pred,Sale_Price_test, len(vars_to_train)).loc['MSE'].values[0])
    r2.append(get_metrics(pred,Sale_Price_test, len(vars_to_train)).loc['R2Adj'].values[0])
    v.append(vars_to_train)
    

# results = pd.DataFrame() >> mutate (MSE = mse_, P= p, used_vars = v , R_adj = r2)
# results.to_csv('model/Random_forest/importance_tunning_final.csv')
results = pd.read_csv('model/Random_forest/importance_tunning_final.csv')
(
    results 
        >> pivot_longer(cols=['MSE','R_adj'], names_to='metric', values_to='Score' )
        >> ggplot (aes (x = 'P',y= 'Score' ))
        +  geom_point()
        + facet_wrap("~metric",ncol = 1, scales  = 'free_y')
)

selected_vars2 = (results >> top_n(1, _.R_adj)>> pull("used_vars"))[0]

step_select =ColumnTransformer(
  [('selector', 'passthrough', selected_vars2 )],
  verbose_feature_names_out = False,
  remainder='drop').set_output(transform = "pandas")
# ['Area_Per_Room_c_', 'Garage_Area', '1st_Flr_SF', '1s_Floor_prop_c_', 'Kitchen_Qual_TA_x_Kitchen_AbvGr', 'Total_Bsmt_SF', 'Gr_Liv_A/Loot_A_c__x_Gr_Liv_A/Loot_A_c_', 'Neighborhood_NridgHt_x_Gr_Liv_Area', 'Gr_Liv_A/Loot_A_c_', 'Garage_Yr_Blt', 'Garage_Qual_TA_x_Garage_Prop_overGLA_c_', 'Gr_Liv_Area', 'Bsmt_Qual_TA_x_Bsmt_Prop_c_', 'Year_Remod/Add', 'Last_remod_c_', 'Garage_Prop_overGLA_c_', 'Antique2_c_', 'Bsmt_Unf_SF', 'Antique_c_', 'BsmtFin_SF_1', 'Year_Built', 'Overall_Qual_Good_or_Excellent_x_1s_Floor_prop_c_', 'Bsmt_Prop_c_']
rf_pipline = Pipeline([
    ('preprocessor', get_df),
    ('step_select', step_select),
    ('regressor', rf2)
])

rf_pipline.fit(ames_x_train, ames_y_train)
"""
        R E S U L T A D O S 
              F I N A L E S
                    T E S T 
                
"""



y_pred = rf_pipline.predict(ames_x_test) * ames_x_test.Gr_Liv_Area

get_metrics(y_pred, Sale_Price_test, len (selected_vars2))



"""
         R E S U L T A D O S 
                 F I N A L E S
                     V A L I D A T I O N 
"""

y_pred = rf_pipline.predict(ames_x_val) * ames_x_val.Gr_Liv_Area

get_metrics(y_pred, Sale_Price_validation, len (selected_vars2))
