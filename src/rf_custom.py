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


 ## SELECCIÓN DE VARIABLES

cat_cols = ['MS_SubClass', 
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
                # 'Mo_Sold', 
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
            # 'Year_Sold',
            'area_per_car',
            'last_remod',
            'wood_prop']  



# Definir las métricas de desempeño que deseas calcular como funciones de puntuación

def adjusted_r2_score(y_true, y_pred, n, p):
  r2 = r2_score(y_true, y_pred)
  adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
  return adjusted_r2





prep = ColumnTransformer([
    ('dummy', OneHotEncoder(drop = 'first',handle_unknown  = 'ignore', sparse_output = False), cat_cols),
    ('selector' , 'passthrough', num_cols)],
    remainder = 'drop',
    verbose_feature_names_out = False).set_output(transform = 'pandas')
    
t_df=prep.fit_transform(ames_x_train)
# Definir el objeto K-Fold Cross Validator
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

param_grid = {
 'max_depth': range(5, 10),
 'min_samples_split': range(3, 10),
 'min_samples_leaf': range(3,10),
 'max_features': range(13, 57,5)
}




scoring = {
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    'r2': make_scorer(adjusted_r2_score, greater_is_better=True, n=np.ceil(len(ames_x_train)), p=len(t_df.columns)),
    'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
    'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
}




    
    
pipeline = Pipeline([
    ('preprocessor', prep),
    ('regressor', GridSearchCV(
      RandomForestRegressor(), 
      param_grid = param_grid, 
      cv=kf, 
      scoring=scoring, 
      refit='neg_mean_squared_error',
      verbose=2, 
      n_jobs=-1,
      error_score='raise')
     )
])

pipeline.fit(ames_x_train, Sale_Price_train)

pd.to_pickle(pipeline, 'grid_search_random_forest2.pkl')

CV_results = pd.DataFrame(pipeline.named_steps['regressor'].cv_results_)


pd.set_option('display.max_columns', 500)
CV_results.columns



# Puedes seleccionar las columnas de interés, por ejemplo:

summary_df = (
  CV_results >>
  select(-_.contains("split._"), -_.contains("time"), -_.params)
)
summary_df


(
  summary_df >> filter ( _.mean_test_r2>0.8)>>
  select(_.param_max_depth, _.param_max_features, _.param_min_samples_leaf, 
         _.param_min_samples_split, _.mean_test_r2) >>
  pivot_longer(
    cols = ["param_max_depth", "param_max_features", "param_min_samples_leaf", "param_min_samples_split"],
    names_to="parameter",
    values_to="value") >>
  ggplot(aes(x = "value", y = "mean_test_r2")) +
  geom_point(size = 0.7, alpha = 0.5) +
  facet_wrap("~parameter", scales = "free_x") +
  xlab("Parameter value") +
  ylab("R^2 promedio") +
  ggtitle("Parametrización de Random Forest vs R^2")
)


(
  summary_df >>filter ( _.mean_test_r2>0.8)>>
  ggplot(aes(x = "param_max_features", y = "mean_test_r2", color = "param_max_depth")) +
  geom_point(alpha = 0.4) +
  ggtitle("Parametrización de Random Forest vs R^2") +
  xlab("Parámetro: Número de features por árbol") +
  ylab("R^2 promedio")
)


voting = VotingRegressor([
  ('lm', pipeline_linreg),
  ('rf',rf_pipline)],
  weights = [0.75,0.25])



voting.fit(ames_x_train, ames_y_train)


y_pred = voting.predict(ames_x_test) *ames_x_test.Gr_Liv_Area
y_obs = Sale_Price_test
get_metrics(y_pred, y_obs, 67)
(
  ggplot (aes(x=y_pred, y=y_obs))
  + geom_point()
  + geom_abline(intercept = 0, slope =1)
)
