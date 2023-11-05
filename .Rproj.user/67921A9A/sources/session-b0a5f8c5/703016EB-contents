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

cat_columns = ['MS_SubClass', 
                'MS_Zoning',
                'Street', 
                'Alley', 
                'Lot_Shape',
                'Land_Contour', 
                'Utilities', 
                'Lot_Config',
                'Land_Slope', 
                'Neighborhood',
                'Condition_1',
                'Condition_2', 
                'Bldg_Type', 
                'House_Style',
                'Overall_Cond', 
                'Roof_Style', 
                'Roof_Matl',
                'Exterior_1st',
                'Exterior_2nd', 
                'Mas_Vnr_Type', 
                'Exter_Cond',
                'Foundation', 
                'Bsmt_Cond',
                'Bsmt_Exposure', 
                'BsmtFin_Type_1', 
                'BsmtFin_Type_2', 
                'Heating',
                'Heating_QC', 
                'Central_Air', 
                'Electrical', 
                'Functional', 
                'Garage_Type',
                'Garage_Finish', 
                'Garage_Cond', 
                'Paved_Drive', 
                'Pool_QC', 
                'Fence',
                'Misc_Feature',
                'Mo_Sold', 
                # 'Sale_Type', 
                # 'Sale_Condition'
                ]

num_cols =[ 'Bedroom_AbvGr',
            'BsmtFin_SF_1',
            'BsmtFin_SF_2',
            'Bsmt_Full_Bath',
            'Bsmt_Half_Bath',
            'Bsmt_Unf_SF',
            'Enclosed_Porch',
            'Fireplaces',
            'First_Flr_SF',
            'Full_Bath',
            'Garage_Area',
            'Garage_Cars',
            'Gr_Liv_Area',
            'Half_Bath',
            'Kitchen_AbvGr',
            'Latitude',
            'Longitude',
            'Lot_Area',
            'Lot_Frontage',
            'Mas_Vnr_Area',
            'Misc_Val',
            'Open_Porch_SF',
            'Pool_Area',
            'Screen_Porch',
            'Second_Flr_SF',
            'Three_season_porch',
            'TotRms_AbvGrd',
            'Total_Bsmt_SF',
            'Wood_Deck_SF',
            'Year_Built',
            'Year_Remod_Add',
            'Year_Sold',
            'area_per_car',
            'last_remod',
            'wood_prop']  

def adjusted_r2_score(y_true, y_pred, n, p):
  r2 = r2_score(y_true, y_pred)
  adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
  return adjusted_r2

###JUST NUMERIC KNN

preprocessor= ColumnTransformer(
    transformers = [
        ("selector", "passthrough", num_cols),
        ('scaler', StandardScaler(), num_cols),
    ],
    verbose_feature_names_out = False,
    remainder = 'drop'  # Mantener las columnas restantes sin cambios
    ).set_output(transform="pandas")
  
#####   C R O S S   V A L I D A T I O N   #####
##### Just numeric KNN 

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

param_grid = {
 'n_neighbors': range(5, 26),
 'weights': ['distance'],
 'metric': ['minkowski'],
 'p': range(3,10)
  }

scoring = {
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    'r2': make_scorer(adjusted_r2_score, greater_is_better=True, n=np.ceil(len(ames_x_train)), p=len(num_cols)),
    'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
    'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
}

pipeline_knn_numeric_cv = Pipeline(
    [('prep', preprocessor),
    ('column_selection', column_selector),
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

# pipeline_knn_numeric_cv.fit(ames_x_train, ames_y_train)

# results_cv = pipeline_knn_numeric_cv.named_steps['regressor'].cv_results_



with open('data/aames_cv_numeric.pkl', 'rb') as f:
    results_cv_NUM_KNN = pickle.load(f)


# Convierte los resultados en un DataFrame
pd.set_option('display.max_columns', 500)
results_df_numeric = pd.DataFrame(results_cv_NUM_KNN)

summary_num_cv = (
  results_df_numeric >>
  select(-_.contains("split"), -_.contains("time"), -_.params)
)

  
(
  summary_num_cv >> filter (_.param_p==3) >> 
  mutate(min_err=_.mean_test_r2-_.std_test_r2,max_err=_.mean_test_r2+_.std_test_r2 ) >>
  ggplot(aes(x = "param_n_neighbors", y = "mean_test_r2", 
             color =  "param_p", shape = "param_weights")) +
  geom_point(alpha = 0.65) +
  geom_errorbar(aes(ymin='min_err', ymax='max_err'), show_legend=False) +
  ggtitle("Parametrización de KNN vs R^2") +
  xlab("Parámetro: Número de vecinos cercanos") +
  ylab("R^2 promedio")
)

knn_numeric_model = Pipeline(
    [('prep_1', preprocessor),
    ('regressor',  KNeighborsRegressor(
                              n_neighbors=7, 
                              weights = 'distance',
                              metric = 'minkowski', 
                              p=3))
     ]
)

knn_numeric_model.fit(ames_x_train, ames_y_train)

y_obs = ames_y_test * ames_x_test.Gr_Liv_Area
y_pred = knn_numeric_model.predict(ames_x_test) *ames_x_test.Gr_Liv_Area

pd.DataFrame(get_metrics(y_pred,y_obs, len(num_cols)))


######## CATEGORICAL KNN 
OHE=ColumnTransformer([
  ('droper', 'drop', num_cols + ['Sale_Type', 'Sale_Condition']),
  ('ohe', OneHotEncoder(drop='first',handle_unknown='ignore' , sparse_output=False, min_frequency=20),cat_columns )
  ],
    verbose_feature_names_out = False,
    remainder = 'passthrough'  # Mantener las columnas restantes sin cambios
    ).set_output(transform="pandas")


param_grid = {
 'n_neighbors': range(0, 105, 5),
 'weights': ['uniform', 'distance'],
 'metric': ['minkowski','euclidean', 'manhattan', 'hamming', 'cosine', 'chebyshev'],
 'p': range(3,10)
}
    
    
    
    
pipeline_knn_cats_cv = Pipeline(
    [('ohe', OHE),
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
pipeline_knn_cats_cv.fit(ames_x_train, ames_y_train)

reuslts_cat_knn = pipeline_knn_cats_cv.named_steps['regressor'].cv_results_

results_cat= pd.DataFrame(reuslts_cat_knn)


summary_cat = (
  results_cat >>
  select(-_.contains("split"), -_.contains("time"), -_.params)
)



(
  summary_cat >> 
  filter (_.param_metric !='chebyshev',
          _.param_n_neighbors <50,
          _.param_weights == 'distance') >>
  ggplot(aes(x = "param_n_neighbors", y = "mean_test_r2", size = "param_p",
             color = "param_metric")) +
  geom_point(alpha = 0.65) +
  ggtitle("Parametrización de KNN vs R^2") +
  xlab("Parámetro: Número de vecinos cercanos") +
  ylab("R^2 promedio")
)




##################  Second KNN  ###############################

param_grid = {
 'n_neighbors': range(0, 30),
 'weights': ['distance'],
 'metric': ['euclidean', 'manhattan', 'hamming', 'cosine', ]
}
    
pipeline_knn_cats_cv_2nd = Pipeline(
    [('ohe', OHE),
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

pipeline_knn_cats_cv_2nd.fit(ames_x_train, ames_y_train)


reuslts_cat_knn_f = pipeline_knn_cats_cv_2nd.named_steps['regressor'].cv_results_


# with open('src/ames_knn_last_cv.pkl', 'wb') as f:
#     pickle.dump(results_cv, f, pickle.HIGHEST_PROTOCOL)

results_cat_df= pd.DataFrame(reuslts_cat_knn_f)
 

summary_cat_f = (
  results_cat_df >>
  select(-_.contains("split"), -_.contains("time"), -_.params)
)



(
  summary_cat_f >> filter(_.param_n_neighbors>7, _.param_metric == 'manhattan')>>
   mutate(min_err=_.mean_test_r2-_.std_test_r2,max_err=_.mean_test_r2+_.std_test_r2 ) >>
  ggplot(aes(x = "param_n_neighbors", 
              y = "mean_test_r2",
             color = "param_metric")) +
  geom_point(position=position_dodge(width=0.7)) +
  geom_errorbar(aes(ymin='min_err', ymax='max_err'),
                  position=position_dodge(width=0.7),show_legend=False) +
  ggtitle("Parametrización de KNN vs R^2") +
  # ylim([0.5,0.7])+
  xlab("Parámetro: Número de vecinos cercanos") +
  ylab("R^2 promedio")
)

######### FINAL KNN MODEL  ###########  



final_knn_cat= Pipeline(
    [('ohe', OHE),
    ('regressor', KNeighborsRegressor(n_neighbors =14 ,
                        weights = 'distance',
                        metric = 'manhattan')
     )]
)

final_knn_cat.fit(ames_x_train, ames_y_train)


y_obs = ames_y_test * ames_x_test.Gr_Liv_Area
y_pred = final_knn_cat.predict(ames_x_test) *ames_x_test.Gr_Liv_Area

OHE.fit(ames_x_train)
metrics = pd.DataFrame(get_metrics(y_pred,y_obs, len(list(OHE.get_feature_names_out())) ))

mse=(metrics>> filter(_.Metric== 'MSE') >> pull ('Value'))[0]

importance = np.zeros(ames_x_test[cat_columns].shape[1])

for i in range(ames_x_test[cat_columns].shape[1]):
    ames_x_test_permuted = ames_x_test[cat_columns].copy()
    ames_x_test_permuted.iloc[:, i] = shuffle(ames_x_test_permuted.iloc[:, i], random_state=42)  
    # Permuta una característica
    y_pred_permuted = final_knn_cat.predict(ames_x_test_permuted) * ames_x_test.Gr_Liv_Area
    mse_permuted = mean_squared_error(ames_y_test, y_pred_permuted)
    importance[i] = mse - mse_permuted
    print(i)

# Calcula la importancia relativa
importance = importance / importance.sum()
importance


importance_df = pd.DataFrame({
  'Variable': cat_columns, 
  'Importance': importance
  })

# Crea la gráfica de barras
(
  importance_df >>
  ggplot(aes(x= 'reorder(Variable, Importance)', y='Importance')) + 
  geom_bar(stat='identity', fill='blue', color = "black") + 
  labs(title='Importancia de las Variables', x='Variable', y='Importancia') +
  coord_flip()
)

y_pred = knn_numeric_model.predict(ames_x_test) *ames_x_test.Gr_Liv_Area

metrics = pd.DataFrame(get_metrics(y_pred,y_obs, len(num_cols))




# lm knn model


num_cols =sorted( ['Wood_Deck_SF', 
            'First_Flr_SF',
            'Fireplaces',
            'Open_Porch_SF',
            'Enclosed_Porch',
            'Lot_Area',
            'wood_prop',
            'Total_Bsmt_SF',
            'Garage_Area',
            'Gr_Liv_Area',
            'area_per_car',
            'last_remod',
            'Bsmt_Full_Bath', 
            'Three_season_porch',
            'BsmtFin_SF_1', 
            'BsmtFin_SF_2', 
            'Bsmt_Unf_SF',
            'Bsmt_Half_Bath', 
            'Full_Bath',
            'Kitchen_AbvGr',
            'Half_Bath',
            'Mas_Vnr_Area',
            'Misc_Val',
            'Lot_Frontage',
            'Bedroom_AbvGr','Year_Built','Garage_Cars'])

cat_cols = ['Bldg_Type', 
            'Bsmt_Exposure', 
            'Central_Air','Mo_Sold', 
            'Condition_1', 
            'Condition_2', 
            'Electrical', 
            'Mas_Vnr_Type',
            'Fence', 
            'Exter_Cond',
            'Foundation', 
            'Garage_Finish', 
            'Garage_Type', 
            # 'Misc_Feature', 
            'Paved_Drive', 
            'Heating_QC',
            'Overall_Cond',
            'MS_SubClass', 
            'Bsmt_Cond',
            'BsmtFin_Type_1', 
            'BsmtFin_Type_2', 
            'House_Style',
            'Lot_Shape', #'Roof_Style'
            'Neighborhood']
            
columnas_seleccionadas = num_cols + cat_cols 



preprocessor_1 = ColumnTransformer(
    transformers = [
        ('scaler', StandardScaler(), num_cols),
        ('OHE', OneHotEncoder(drop='first',handle_unknown='ignore' , sparse_output=False, min_frequency=20), cat_cols)
    ],
    verbose_feature_names_out = False,
    remainder = 'drop'  # Mantener las columnas restantes sin cambios
    ).set_output(transform="pandas")





interaction_transformer = PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)

drop_cols=['Neighborhood_Crawford',
            'Neighborhood_infrequent_sklearn',
            'BsmtFin_Type_1_BLQ',
            'Exter_Cond_Good',
            # 'Condition_2_Feedr',
            'Condition_1_RRAe',
            # 'Condition_1_Feedr',
            'Garage_Type_No_Garage',
            # 'Condition_1_PosA',
            'Bsmt_Full_Bath',
            'Bsmt_Half_Bath',
            # 'Condition_2_RRAe',
            'Foundation_CBlock',
            'MS_SubClass_Two_Story_1946_and_Newer',
            'Garage_Finish_No_Garage',
            'Bldg_Type_TwoFmCon',
            'Mo_Sold_5',
            'Paved_Drive_Partial_Pavement',
            'Bsmt_Cond_Typical',
            'Heating_QC_Fair',
            'BsmtFin_Type_1_Unf',
            'Electrical_SBrkr',
            'Fence_Good_Wood',
            # 'Electrical_Unknown',
            'Garage_Type_Detchd',
            'Foundation_Slab',
            'House_Style_infrequent_sklearn',
            'Three_season_porch',
            'BsmtFin_Type_2_GLQ',
            'Mo_Sold_7',
            'Misc_Val',
            'BsmtFin_Type_1_No_Basement',
            'Foundation_PConc',
            'Mo_Sold_12',
            'Mo_Sold_10',
            'BsmtFin_Type_2_No_Basement',
            'last_remod',
            'Bsmt_Cond_No_Basement',
            'Condition_2_infrequent_sklearn',
            'Exter_Cond_Typical',
            'Bldg_Type_OneFam',
            'Mo_Sold_9',
            'Electrical_infrequent_sklearn',
            'Bsmt_Cond_infrequent_sklearn',
            'Bsmt_Cond_Good',
            'Foundation_infrequent_sklearn',
            'Bldg_Type_TwnhsE',
            'Fence_No_Fence']

  

preprocessor_2=ColumnTransformer(
  transformers=[
    ("selector", "drop", drop_cols)
    ('interaction_1', interaction_transformer, ['Lot_Area', 'Gr_Liv_Area']),
    ('interactions2', interaction_transformer, ['Year_Built', 'Overall_Cond_Average']),
    ('interactions3', interaction_transformer, ['Full_Bath', 'Bedroom_AbvGr']),
    ('interactions3.1', interaction_transformer, ['House_Style_SLvl', 'Overall_Cond_Fair']),
    ('interactions3.4', interaction_transformer, ['wood_prop', 'Mas_Vnr_Area']),
    ('interactions3.2', interaction_transformer, ['First_Flr_SF', 'last_remod']),
    ('interactions3.5', interaction_transformer, ['BsmtFin_SF_1', 'BsmtFin_Type_2_Unf']),
    ('interactions4', interaction_transformer, ['Misc_Val', 'Misc_Feature_TenC'])
  ],
  verbose_feature_names_out = False,
  remainder='passthrough'
)


param_grid = {
 'n_neighbors': range(0, 105, 5),
 'weights': ['uniform', 'distance'],
 'metric': ['minkowski','euclidean', 'manhattan', 'hamming', 'cosine', 'chebyshev'],
 'p': range(3,10)
}
    

pipeline_lm_set_cv= Pipeline([
  ('prep1', preprocessor_1),
  ('interactions', preprocessor_2),
  ('regressor', GridSearchCV(
      KNeighborsRegressor(), 
      param_grid, 
      cv=kf, 
      scoring=scoring, 
      refit='neg_mean_squared_error',
      verbose=3, 
      n_jobs=7)
     )])


pipeline_lm_set_cv.fit(ames_x_train, ames_y_train)



results_cv_lmset=pipeline_lm_set_cv.named_steps['regressor'].cv_results_

results_cv_lmset_df= pd.DataFrame(results_cv_lmset)


summary_df = (
  results_cv_lmset_df >>
  select(-_.contains("split"), -_.contains("time"), -_.params)
)



(
  results_cv_lmset_df >> 
  # filter (_.param_metric !='chebyshev',
  #         _.param_n_neighbors <50,
  #         _.param_weights == 'distance') >>
  ggplot(aes(x = "param_n_neighbors", y = "mean_test_r2", size = "param_p",
             color = "param_metric")) +
  geom_point(alpha = 0.65) +
  ggtitle("Parametrización de KNN vs R^2") +
  xlab("Parámetro: Número de vecinos cercanos") +
  ylab("R^2 promedio")
)

param_grid = {
 'n_neighbors': range(3, 25),
 'weights': [ 'distance'],
 'metric': ['manhattan'],
}
    

pipeline_lm_set_cv= Pipeline([
  ('prep1', preprocessor_1),
  ('interactions', preprocessor_2),
  ('regressor', GridSearchCV(
      KNeighborsRegressor(), 
      param_grid, 
      cv=kf, 
      scoring=scoring, 
      refit='neg_mean_squared_error',
      verbose=3, 
      n_jobs=7)
     )])


pipeline_lm_set_cv.fit(ames_x_train, ames_y_train)



results_cv_lmset=pipeline_lm_set_cv.named_steps['regressor'].cv_results_

results_cv_lmset_df= pd.DataFrame(results_cv_lmset)


summary_df = (
  results_cv_lmset_df >>
  select(-_.contains("split"), -_.contains("time"), -_.params)
)



(
  results_cv_lmset_df >> 
  # filter (_.param_metric !='chebyshev',
  #         _.param_n_neighbors <50,
  #         _.param_weights == 'distance') >>
  ggplot(aes(x = "param_n_neighbors", y = "mean_test_r2", 
             color = "param_metric", shape = "param_weights")) +
  geom_point(alpha = 0.65) +
  ggtitle("Parametrización de KNN vs R^2") +
  xlab("Parámetro: Número de vecinos cercanos") +
  ylab("R^2 promedio")
)
best_estimator =pipeline_lm_set_cv.named_steps['regressor'].best_estimator_



pipeline_lm_set_final= Pipeline([
  ('prep1', preprocessor_1),
  ('interactions', preprocessor_2),
  ('regressor', best_estimator)])
  
pipeline_lm_set_final.fit(ames_x_train, ames_y_train)


y_pred = pipeline_lm_set_final.predict(ames_x_test)*ames_x_test.Gr_Liv_Area






pd.DataFrame(
  get_metrics(y_pred, y_obs, predictors= p)
)







voting_reg = VotingRegressor(estimators=[
  ('knn_cat', final_knn_cat), 
  ('knn_num', knn_numeric_model),
  ('knn_lm',pipeline_lm_set_final )
  ], weights = [0.10,0.2,0.7])



voting_reg.fit(ames_x_train, ames_y_train)

y_pred = voting_reg.predict(ames_x_test) * ames_x_test.Gr_Liv_Area

p = len(list(preprocessor_2.get_feature_names_out()))
metrics = pd.DataFrame(get_metrics(y_pred,y_obs,p))
metrics




with open('src/cv_lmset__knn.pkl', 'wb') as f:
    pickle.dump(results_cv_lmset, f, pickle.HIGHEST_PROTOCOL)

with open('src/best_model_lmset__knn.pkl', 'wb') as f:
    pickle.dump(pipeline_lm_set_final, f, pickle.HIGHEST_PROTOCOL)




with open('src/voting_knn.pkl', 'wb') as f:
    pickle.dump(voting_reg, f, pickle.HIGHEST_PROTOCOL)










