from mlxtend.feature_selection import ColumnSelector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from plydata.one_table_verbs import pull
from mizani.formatters import comma_format, dollar_format
from plotnine import *
from siuba import *

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.2f}'.format


def div_columns(X, c1, c2):
    X["c1_c2"] = X[c1]/ X[c2]
    return X

def get_metrics (y_pred, y_obs,predictors):
  me = np.mean(y_obs - y_pred)
  mae = mean_absolute_error(y_obs, y_pred)
  mape = mean_absolute_percentage_error(y_obs, y_pred)
  mse = mean_squared_error(y_obs, y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_obs, y_pred)

  n = len(y_obs)  # Número de observaciones
  p = len(predictors)  # Número de predictores 
  r2_adj = 1 - (n - 1) / (n - p - 1) * (1 - r2)
  

  metrics_data = {
      "Metric": ["ME", "MAE", "MAPE", "MSE", "RMSE", "R^2", "R^2 Adj"],
      "Value": [me, mae, mape, mse, rmse, r2, r2_adj]
  }
  return metrics_data

def rmv_elements (list_of_elements, complete_list):
  for element in list_of_elements:
    complete_list.remove(element)
  return complete_list




#### CARGA DE DATOS ####
ames = pd.read_csv("data/ames.csv") >> mutate (Price_4_GLA= _.Sale_Price/_.Gr_Liv_Area)
print("Tamaño de conjunto completo: ", ames.shape)

y = ames >>  pull("Price_4_GLA")
X = select(ames, -_.Sale_Price, -_.Price_4_GLA)

X=(X >> mutate(wood_prop= _.Wood_Deck_SF/_.Gr_Liv_Area,
              area_per_car=case_when({_.Garage_Cars== 0:0,
                                  _.Garage_Cars== 0:0,
                                  True:_.Garage_Area/ _.Garage_Cars}),
              last_remod = case_when({_.Year_Built>_.Year_Remod_Add: 2023-_.Year_Remod_Add,
                                  True: 2023-_.Year_Built}),
              Mo_Sold= _.Mo_Sold.astype('object')))

numeric_column = ames >> pull("Price_4_GLA")
quartiles = np.percentile(numeric_column, [25, 50, 75])

# Crea una nueva variable categórica basada en los cuartiles
stratify_variable = pd.cut(
 numeric_column, 
 bins=[float('-inf'), quartiles[0], quartiles[1], quartiles[2], float('inf')],
 labels=["Q1", "Q2", "Q3", "Q4"]
 )

ames_x_train, ames_x_test, ames_y_train, ames_y_test = train_test_split(
 X, y, 
 test_size = 0.20, 
 random_state = 12345, 
 stratify = stratify_variable
 )
 


### FEATURE ENGINEERING ####


# Seleccionamos las variales numéricas de interés
num_cols = ['Wood_Deck_SF', 
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
            'BsmtFin_SF_2', 'Bsmt_Unf_SF',
            'Bsmt_Half_Bath', 
            'Full_Bath',
            'Kitchen_AbvGr',
            'Half_Bath',
            'Mas_Vnr_Area',
            'Misc_Val',
            'Lot_Frontage',
            'Bedroom_AbvGr']
            
            
# 'Lot_Frontage', 'Lot_Area', 'Year_Built', 'Year_Remod_Add', 'Mas_Vnr_Area', 'BsmtFin_SF_1', 
# 'BsmtFin_SF_2', 'Bsmt_Unf_SF', 'Total_Bsmt_SF', 'First_Flr_SF', 'Second_Flr_SF', 'Gr_Liv_Area', 
# 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr', 
# 'TotRms_AbvGrd', 'Fireplaces', 'Garage_Cars', 'Garage_Area', 'Wood_Deck_SF', 'Open_Porch_SF', 
# 'Enclosed_Porch', 'Three_season_porch', 'Screen_Porch', 'Pool_Area', 'Misc_Val', 'Mo_Sold', 
# 'Year_Sold', 'Sale_Price', 'Longitude', 'Latitude'
# Seleccionamos las variables categóricas de interés
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
            'Misc_Feature', 
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

# Juntamos todas las variables de interés

columnas_seleccionadas = num_cols +cat_cols_t+cat_cols_4+cat_cols_3+cat_num +['Year_Built','Garage_Cars']
pipe = ColumnSelector(columnas_seleccionadas)
ames_x_train_selected = pipe.fit_transform(ames_x_train)

ames_train_selected = pd.DataFrame(
  ames_x_train_selected, 
  columns = columnas_seleccionadas
  )

ames_test_selected = pd.DataFrame(
  pipe.fit_transform(ames_x_test), 
  columns = columnas_seleccionadas
  )


## TRANSFORMACIÓN DE COLUMNAS

###################################################
############# CUSTOM FUNCTIONS ####################
###################################################


div_transformer = FunctionTransformer(
 custom_function,
 feature_names_out = 'one-to-one',
  kw_args={'c1': 'Column1', 'c2': 'Column2'}
 )
interaction_transformer = PolynomialFeatures(
 degree = 2, 
 interaction_only = True, 
 include_bias = False
 )
 interaction_transformer_wb = PolynomialFeatures(
 degree = 2, 
 interaction_only = True, 
 include_bias = False
 )
# ColumnTransformer para aplicar transformaciones
from sklearn.impute import SimpleImputer


preprocessor_1 = ColumnTransformer(
    transformers = [
        # ('first_selection', ColumnSelector(), columnas_seleccionadas),
        ('scaler', StandardScaler(), num_cols),
        ('imputer',  SimpleImputer(missing_values=np.nan, strategy='mean'), ['Year_Built','Garage_Cars']),
        ('OHE_total', OneHotEncoder(drop='first',handle_unknown='ignore' , sparse_output=False), cat_cols_t),
        ('OHE_total1', OneHotEncoder(drop='first',handle_unknown='ignore' ,sparse_output=False,min_frequency =25), cat_cols_4),
        ('OHE_total2', OneHotEncoder(drop='first',handle_unknown='ignore' ,sparse_output=False,min_frequency =25), cat_cols_3),
        ('OHE_total3', OneHotEncoder(drop='first',handle_unknown='ignore' ,sparse_output=False,min_frequency  =25), cat_num)
        # ('select','drop',drop_cols)
    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough'  # Mantener las columnas restantes sin cambios
)

ames_x_train_trans= pd.DataFrame(
  preprocessor_1.fit_transform(ames_train_selected),
  columns=preprocessor_1.get_feature_names_out()
  )
  
ames_x_test_trans = pd.DataFrame(
  preprocessor_1.transform(ames_test_selected),
  columns=preprocessor_1.get_feature_names_out()
  )

drop_cols=['Neighborhood_Crawford',
            'Neighborhood_infrequent_sklearn',
            'House_Style_Two_Story',
            'Foundation_Wood',
            'Condition_2_Feedr',
            'Condition_1_RRAe',
            'Condition_1_Feedr',
            'Garage_Type_No_Garage',
            'Condition_1_PosA',
            'Bsmt_Full_Bath',
            'Bsmt_Half_Bath',
            'Condition_2_RRAe',
            'Foundation_CBlock',
            'MS_SubClass_Two_Story_1946_and_Newer',
            'Garage_Finish_No_Garage',
            'Bldg_Type_TwoFmCon',
            'Mo_Sold_5',
            'Paved_Drive_Partial_Pavement',
            'Bsmt_Cond_Typical',
            'Condition_1_RRNe',
            'BsmtFin_Type_1_Unf',
            'Electrical_SBrkr',
            'Fence_Good_Wood',
            'Electrical_Unknown',
            'Garage_Type_Detchd',
            'Foundation_Slab',
            # 'House_Style_infrequent_sklearn',
            'Three_season_porch',
            'Fence_Minimum_Wood_Wire',
            'Mo_Sold_7',
            'BsmtFin_Type_1_No_Basement',
            'Foundation_PConc',
            'Mo_Sold_12',
            'Mo_Sold_10',
            'BsmtFin_Type_2_No_Basement',
            'Electrical_FuseP',
            'Bsmt_Cond_No_Basement',
            'Condition_1_RRNn',
            'Exter_Cond_Typical',
            'Exter_Cond_Good',
            'Garage_Type_CarPort',
            'Electrical_Mix',
            'Bsmt_Cond_infrequent_sklearn',
            'Fence_No_Fence']

  

preprocessor_2=ColumnTransformer(
  transformers=[
    ("selector", "drop", drop_cols),
    ('interaction_1', interaction_transformer_wb, ['Lot_Area', 'Gr_Liv_Area']),
    ('interactions2', interaction_transformer, ['Year_Built', 'Overall_Cond_Average']),
    ('interactions3', interaction_transformer, ['Full_Bath', 'Bedroom_AbvGr']),
    ('interactions3.1', interaction_transformer, ['House_Style_SLvl', 'Overall_Cond_Fair']),
    ('interactions3.2', interaction_transformer, ['Kitchen_AbvGr', 'last_remod']),
    ('interactions4', interaction_transformer, ['Misc_Val', 'Misc_Feature_TenC'])
    # ('div', FunctionTransformer(
    #               div_columns,
    #               feature_names_out = 'one-to-one',
    #               kw_args={'c1': 'Gr_Liv_Area', 'c2': 'Lot_Area'}), ['Gr_Liv_Area', 'Lot_Area'])
  ],
  verbose_feature_names_out = False,
  remainder='passthrough'
)

preprocessor_2.fit(ames_x_train_trans)
# new_interactions= ['Lot_div_Gross']
feature_names = list(preprocessor_2.get_feature_names_out())
# feature_names.append(new_interactions)

  
##### Extracción de coeficientes
transformed_df = pd.DataFrame(
  preprocessor_2.fit_transform(ames_x_train_trans),
  columns=feature_names
  )
transformed_df.info()

X_train_with_intercept = sm.add_constant(transformed_df)
model = sm.OLS(ames_y_train, X_train_with_intercept).fit()

model.summary()


pipeline =Pipeline([
  ('prep2',preprocessor_2),
  ('regressor', LinearRegression()),
  
])   

# Entrenar el pipeline
results = pipeline.fit(ames_x_train_trans, ames_y_train)

## PREDICCIONES
y_pred = pipeline.predict(ames_x_test_trans)

ames_test = (
  ames_x_test >>
  mutate(Sale_Price_Pred = y_pred*_.Gr_Liv_Area, Sale_Price =ames_y_test*_.Gr_Liv_Area)
)


(
ames_test >>
  select(_.Sale_Price, _.Sale_Price_Pred)
)

##### Métricas de desempeño

y_obs = ames_test["Sale_Price"]
y_pred = ames_test["Sale_Price_Pred"]


mtrcs_dt=get_metrics(y_pred, y_obs,predictors=columnas_seleccionadas)


metrics_df = pd.DataFrame(mtrcs_dt)
metrics_df

#### Gráficos de desempeño de modelo

(
    ggplot(aes(x = y_obs, y = y_pred)) +
    geom_point(alpha=0.5) +
    scale_y_continuous(labels = dollar_format(digits=0, big_mark=','), limits = [0, 600000] ) +
    scale_x_continuous(labels = dollar_format(digits=0, big_mark=','), limits = [0, 600000]) +
    geom_abline(color = "red") +
    coord_equal() +
    labs(
      title = "Comparación entre predicción y observación",
      x = "Predicción",
      y = "Observación")
)


(
ames_test >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(x = "error")) +
  geom_histogram(color = "white", fill = "black") +
  geom_vline(xintercept = 0, color = "red") +
  scale_x_continuous(labels=dollar_format(big_mark=',', digits=0)) + 
  ylab("Conteos de clase") + xlab("Errores") +
  ggtitle("Distribución de error")
)


(
ames_test >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(sample = "error")) +
  geom_qq(alpha = 0.3) + stat_qq_line(color = "red") +
  scale_y_continuous(labels=dollar_format(big_mark=',', digits = 0)) + 
  xlab("Distribución normal") + ylab("Distribución de errores") +
  ggtitle("QQ-Plot")
)


(
ames_test >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(x = "Sale_Price")) +
  geom_linerange(aes(ymin = 0, ymax = "error"), colour = "purple") +
  geom_point(aes(y = "error"), size = 0.05, alpha = 0.5) +
  geom_abline(intercept = 0, slope = 0) +
  scale_x_continuous(labels=dollar_format(big_mark=',', digits=0)) + 
  scale_y_continuous(labels=dollar_format(big_mark=',', digits=0)) +
  xlab("Precio real") + ylab("Error de estimación") +
  ggtitle("Relación entre error y precio de venta")
)

#### Validación cruzada ####

# Definir el objeto K-Fold Cross Validator
    
