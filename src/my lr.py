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
            '1st_Flr_SF',
            'Open_Porch_SF',
            'Enclosed_Porch',
            'Lot_Area',
            # 'Wood_prop',
            'Total_Bsmt_SF',
            'Garage_Area',
            'Gr_Liv_Area',
            'Bsmt_Full_Bath', 
            '3Ssn_Porch',
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
            'Central_Air',
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
    ('OPP_prop',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Open_Porch_SF', 'c2': 'Lot_Area', 'feature_name' : 'OPP_prop'}
                                 ),['Open_Porch_SF', 'Lot_Area'] ),
    ('Gr_Liv_Area / Loot_Area',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Gr_Liv_Area', 'c2': 'Lot_Area', 'feature_name' : 'Gr_Liv_Area / Loot_Area'}
                                 ),['Gr_Liv_Area', 'Lot_Area'] ),
    ('Garage_Prop_overGLA',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Garage_Area', 'c2': 'Gr_Liv_Area', 'feature_name' : 'Garage_Prop'}
                                 ),['Garage_Area', 'Gr_Liv_Area'] ),
    ('just_select', 'passthrough', ['Total_Bsmt_SF', 'Open_Porch_SF', 'Garage_Area', 'Lot_Area', 'Gr_Liv_Area', 'Wood_Deck_SF' ])],
    
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

intersection(r_cols, drop_cols)
drop_cols=['Bsmt_Full_Bath',
            'Bsmt_Half_Bath',
            'Misc_Val',
            'Bldg_Type_TwnhsE',
            'Condition_1_Feedr',
            'Condition_2_infrequent_sklearn',
            'Electrical_SBrkr',
            'Electrical_infrequent_sklearn',
            'Mas_Vnr_Type_nan',
            'Fence_infrequent_sklearn',
            'Foundation_CBlock',
            'Foundation_PConc',
            'Foundation_infrequent_sklearn',
            'Garage_Type_Detchd',
            'Bsmt_Cond_infrequent_sklearn',
            'BsmtFin_Type_1_BLQ',
            'BsmtFin_Type_1_Unf',
            'House_Style_SFoyer',
            'House_Style_infrequent_sklearn',
            'Neighborhood_infrequent_sklearn',
            'Misc_Feature_nan']

  

preprocessor_2=ColumnTransformer(
  transformers=[
    ("selector", "drop", drop_cols)
    # ('interaction_1', interaction_transformer_wb, ['Lot_Area', 'Gr_Liv_Area']),
    # ('interactions2', interaction_transformer, ['Year_Built', 'Overall_Cond_Average']),
    # ('interactions3', interaction_transformer, ['Garage_Area', 'Bedroom_AbvGr']),
    # ('interactions3.1', interaction_transformer, ['House_Style_SLvl', 'Overall_Cond_Fair']),
    # ('interactions3.4', interaction_transformer, ['Wood_propGLA_c_', 'Mas_Vnr_Area']),
    # ('interactions3.2', interaction_transformer, ['Garage_Type_Basment', 'BsmtFin_SF_1']),
    # ('interactions3.5', interaction_transformer, ['BsmtFin_SF_1', '1st_Flr_SF']),
    # ('interactions4', interaction_transformer, ['Misc_Val', 'Misc_Feature_TenC'])
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
p_vals = (
  model.pvalues.reset_index() 
    >> select( _.variable == _.index, _.p_val == -1)
    >> mutate (s = case_when({ 
          _.p_val<0.025:'***',
          _.p_val<0.05:'**',
          _.p_val<0.1:'*',
          True:''})
          )
    >> arrange (_.s,- _.p_val)
)

p_vals >> filter (p_vals.variable.str.contains('_c_'))
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
