from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
import statsmodels.api as sm


########   I M  O R T A N C E   MI AND F-TEST   


f_test, _ = f_regression(ames_x_train.select_dtypes(exclude = 'object'), ames_y_train)
f_test /= np.max(f_test)

mi = mutual_info_regression(ames_x_train.select_dtypes(exclude = 'object'), ames_y_train)
mi /= np.max(mi)




columns = ames_x_train.select_dtypes(exclude = 'object').columns.to_list()

importance = pd.DataFrame ({'Variable': columns, 
                'F_test': f_test,
                'MI': mi})

(
    importance 
            >> pivot_longer(cols=['F_test', 'MI'], names_to='metric', values_to='value')
            >> ggplot(aes (x= 'reorder(Variable,value)', y= 'value', fill = 'metric' ) )
            +  geom_col(show_legend = False)
            + facet_wrap('~metric' )
            + coord_flip()
)


#############################################################

pd.options.display.float_format = '{:.8f}'.format


num_cols =['Wood_Deck_SF', 
            'First_Flr_SF',
            'Open_Porch_SF',
            'Enclosed_Porch',
            'Lot_Area',
            # 'Wood_prop',
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
            'Kitchen_AbvGr',
            'Mas_Vnr_Area',
            'Misc_Val',
            'Lot_Frontage',
            'Bedroom_AbvGr']

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

columnas_seleccionadas = num_cols + cat_cols + ['Full_Bath','Year_Built','Fireplaces','Half_Bath','Garage_Cars','Misc_Feature']
 
######################################

def div_columns(X, c1, c2, feature_name ):
    name = feature_name+'_c_'
    X[name] = X[c1]/ X[c2]
    return X[[name]]
  
  
featur_eng = ColumnTransformer(
  [('Wood_Prop',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Wood_Deck_SF', 'c2': 'Lot_Area', 'feature_name' : 'Wood_Prop'}
                                 ), ['Wood_Deck_SF', 'Lot_Area']),
    ('Wood_propGLA',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Wood_Deck_SF', 'c2': 'Gr_Liv_Area', 'feature_name' : 'Wood_propGLA'}
                                 ), ['Wood_Deck_SF', 'Gr_Liv_Area']),
    ('Bsmt_Prop',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Total_Bsmt_SF', 'c2': 'Lot_Area', 'feature_name' : 'Bsmt_Prop'}
                                 ),['Total_Bsmt_SF', 'Lot_Area'] ),
    ('just_select', 'passthrough', ['Total_Bsmt_SF', 'Lot_Area','Gr_Liv_Area', 'Wood_Deck_SF' ])],
    
    verbose_feature_names_out = False,
    remainder = 'passthrough').set_output(transform = 'pandas') 



my_features = (featur_eng.fit_transform(ames_x_train) >> select (_.contains('_c_'))).columns.to_list()

preprocessor_1 = ColumnTransformer(
    transformers = [
        ('scaler', StandardScaler(), num_cols),
        ('just_select', 'passthrough', my_features),
        ('imputer',  SimpleImputer(missing_values=np.nan, strategy='mean'), ['Year_Built','Half_Bath','Fireplaces','Full_Bath','Garage_Cars']),
        ('OHE', OneHotEncoder(drop='first',handle_unknown='ignore' , sparse_output=False, min_frequency=20), cat_cols),
        ('OHE_1', OneHotEncoder(drop='first',handle_unknown='ignore' , sparse_output=False), ['Misc_Feature'])
        
    ],
    verbose_feature_names_out = False,
    remainder = 'drop'  
    ).set_output(transform = 'pandas') 






interaction_transformer = PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)
interaction_transformer_wb = PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)


drop_cols=['Neighborhood_Crawford',
            'Neighborhood_infrequent_sklearn',
            'BsmtFin_Type_1_BLQ',
            'Exter_Cond_Good',
            'Neighborhood_Northridge',
            'Condition_1_RRAe',
            'Electrical_FuseF',
            'Misc_Val',
            'Garage_Type_No_Garage',
            'Condition_1_Feedr',
            'Bsmt_Full_Bath',
            'Bsmt_Half_Bath',
            'Fence_infrequent_sklearn',
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
            'Mo_Sold_4',
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
            'Misc_Val',
            'Bldg_Type_OneFam',
            'Mo_Sold_9',
            'House_Style_SFoyer',
            'Misc_Feature_nan',
            'Electrical_infrequent_sklearn',
            'Bsmt_Cond_infrequent_sklearn',
            'House_Style_One_and_Half_Fin',
            'Bsmt_Cond_Good',
            'area_per_car',
            'Mas_Vnr_Type_nan',
            'Foundation_infrequent_sklearn',
            'Bldg_Type_TwnhsE',
            'Fence_No_Fence']

  

preprocessor_2=ColumnTransformer(
  transformers=[
    ("selector", "drop", drop_cols),
    ('interaction_1', interaction_transformer_wb, ['Lot_Area', 'Gr_Liv_Area']),
    ('interactions2', interaction_transformer, ['Year_Built', 'Overall_Cond_Average']),
    ('interactions3', interaction_transformer, ['Garage_Area', 'Bedroom_AbvGr']),
    # ('interactions3.1', interaction_transformer, ['House_Style_SLvl', 'Overall_Cond_Fair']),
    ('interactions3.4', interaction_transformer, ['Wood_propGLA_c_', 'Mas_Vnr_Area']),
    # ('interactions3.2', interaction_transformer, ['Garage_Type_Basment', 'BsmtFin_SF_1']),
    ('interactions3.5', interaction_transformer, ['BsmtFin_SF_1', 'First_Flr_SF']),
    ('interactions4', interaction_transformer, ['Misc_Val', 'Misc_Feature_TenC'])
  ],
  verbose_feature_names_out = False,
  remainder='passthrough'
).set_output(transform = 'pandas')

pipeline = Pipeline([
  ('feat_eng', featur_eng),
  ('preprocessor', preprocessor_1),
  ('select_interac', preprocessor_2),
  ('regressor', LinearRegression())])

# Entrenar el pipeline
pipeline.fit(ames_x_train, ames_y_train)  
# 
# fitt={'variable' : ['intercept']+list(pipeline.named_steps['regressor'].feature_names_in_),
# 'coef' :[pipeline.named_steps['regressor'].intercept_ ]+list(pipeline.named_steps['regressor'].coef_)}
# coefs_reg =pd.DataFrame(fitt)
#   
# (
#     coefs_reg 
#         >> filter (_.variable != 'intercept')
#         >> mutate( av = abs(_.coef) , m= _.av.max() , coef2=_.coef/_.m)
#         >> top_n(10, abs(_.coef2) )
#         >> ggplot(aes(x='reorder(variable,coef2)', y='coef'))
#         + geom_col()
#         + coord_flip()
#         # 
# )
# 
# (
#     coefs_reg 
#         >> filter (_.variable != 'intercept')
#         >> mutate( av = abs(_.coef) , m= _.av.max() , coef2=_.coef/_.m)
#         >> top_n(-10, abs(_.coef2) )
#         >> ggplot(aes(x='reorder(variable,coef2)', y='coef'))
#         + geom_col()
#         + coord_flip()
#         # 
# )
# abs(-10)
##### Extracción de coeficientes
transformed_df = pipeline.named_steps['select_interac'].transform(pipeline.named_steps['preprocessor'].transform(pipeline.named_steps['feat_eng'].transform(ames_x_train)))

X_train_with_intercept = sm.add_constant(transformed_df)
model = sm.OLS(ames_y_train, X_train_with_intercept).fit()
(
  model.pvalues.reset_index() 
    >> select( _.var == _.index, _.p_val == -1)
    >> mutate (s = case_when({ 
          _.p_val<0.025:'***',
          _.p_val<0.05:'**',
          _.p_val<0.1:'*',
          True:''})
          )
    >> arrange (_.s,- _.p_val)
)
model.summary()

## PREDICCIONES
y_pred = pipeline.predict(ames_x_test) * ames_x_test.Gr_Liv_Area

ames_test = (
  ames_x_test >> mutate(
                  Sale_Price_Pred = y_pred, 
                  Sale_Price =Sale_Price_test
                  )
              >> select (_.contains('Sale_Price'))
)

##### Métricas de desempeño

y_obs = ames_test["Sale_Price"]
y_pred = ames_test["Sale_Price_Pred"]

predictores = transformed_df.columns.to_list()
mtrcs_df =get_metrics(y_pred, y_obs,predictors= len(transformed_df.columns ))


print(mtrcs_df)
# 
# y_pred = pipeline.predict(ames_x_val)*ames_x_val.Gr_Liv_Area
# 
# get_metrics(y_pred,Sale_Price_validation,len(transformed_df.columns ) )
