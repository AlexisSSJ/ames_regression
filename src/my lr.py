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
ames.info()

f_test, _ = f_regression(t.select_dtypes(exclude = 'object'), ames_y_train)
f_test /= np.max(f_test)

mi = mutual_info_regression(t.select_dtypes(exclude = 'object'), ames_y_train)
mi /= np.max(mi)




columns = t.select_dtypes(exclude = 'object').columns.to_list()

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

pd.options.display.float_format = '{:.5f}'.format


num_cols =['Wood_Deck_SF', 
            '1st_Flr_SF',
            'Open_Porch_SF',
            'Enclosed_Porch',
            'Lot_Area',
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
            'Misc_Feature',
            'Paved_Drive', 
            'Heating_QC',
            'Overall_Cond',
            'MS_SubClass', 
            'Bsmt_Cond',
            'BsmtFin_Type_1', 
            'BsmtFin_Type_2', 
            'House_Style',
            'Lot_Shape', #
            'Roof_Style',
            'Neighborhood']
            


 
######################################

def div_columns(X, c1, c2, feature_name ):
    name = feature_name+'_c_'
    X[c1] = X[c1].astype(float)
    X[c2] = X[c2].astype(float)
    X[name] = X[c1]/ X[c2] 
    return X[[name]]


def div_columns2(X, c1, c2, feature_name ):
    name = feature_name+'_c_'
    X[c1] = X[c1].astype(float)
    X[c2] = X[c2].astype(float)
    X[name] = X[c1]/ X[c2] 
    return X[[name]].fillna(0)

def collapse(X, c1, dic):
  return X[[c1]].replace(dic)

def cross_interactions (X, num, cat, return_inputs = 'none'):
    return_cols = []
    df = X.filter(regex=(cat)).copy()
    for column in df.columns.to_list():
      name = column + '_x_'+ num
      X[name] = df[column] * X[num]
      return_cols.append(name)
    if return_inputs == 'none':
      return X[return_cols]
    elif return_inputs == 'both':
      return X[return_cols + df.columns.to_list() + [num]]
    elif return_inputs == 'num':
      return X[return_cols + [num]]
    else:
      return X[return_cols+ df.columns.to_list()]

cond ={4:'bad', 5:'reg',6:'reg', 7:'good', 8:'good', 9:'good'}

feature_eng = ColumnTransformer(
  [('collapse_cond',  FunctionTransformer(
                                 collapse,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Overall_Cond', 'dic' : cond }
                                 ), ['Overall_Cond']),
  ('collapse_cond2',  FunctionTransformer(
                                 collapse,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Overall_Qual', 'dic' : cond }
                                 ), ['Overall_Qual']),
  ('Area_Per_Room',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'TotRms_AbvGrd', 'c2': 'Gr_Liv_Area', 'feature_name' : 'Area_Per_Room'}
                                 ), ['TotRms_AbvGrd', 'Gr_Liv_Area']),
    ('Wood_propGLA',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Wood_Deck_SF', 'c2': 'Gr_Liv_Area', 'feature_name' : 'Wood_propGLA'}
                                 ), ['Wood_Deck_SF', 'Gr_Liv_Area']),
    ('Bsmt_Prop',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Total_Bsmt_SF', 'c2': 'Gr_Liv_Area', 'feature_name' : 'Bsmt_Prop'}
                                 ),['Total_Bsmt_SF', 'Gr_Liv_Area'] ),
    ('Area_p_car',  FunctionTransformer(
                                 div_columns2,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Garage_Area', 'c2': 'Garage_Cars', 'feature_name' : 'Area_p_car'}
                                 ),['Garage_Cars', 'Garage_Area'] ),
    ('Gr_Liv_Area / Loot_Area',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Gr_Liv_Area', 'c2': 'Lot_Area', 'feature_name' : 'Gr_Liv_A/Loot_A'}
                                 ),['Gr_Liv_Area', 'Lot_Area'] ),
    ('Garage_Prop_overGLA',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Garage_Area', 'c2': 'Gr_Liv_Area', 'feature_name' : 'Garage_Prop_overGLA'}
                                 ),['Garage_Area', 'Gr_Liv_Area'] ),
    ('1s_Floor_prop',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': '1st_Flr_SF', 'c2': 'Gr_Liv_Area', 'feature_name' : '1s_Floor_prop'}
                                 ),['1st_Flr_SF', 'Gr_Liv_Area'] ),
    ('just_select', 'passthrough', ['Total_Bsmt_SF', 'TotRms_AbvGrd','Garage_Cars', '1st_Flr_SF', 'Garage_Area', 'Lot_Area', 'Gr_Liv_Area', 'Wood_Deck_SF' ])],
    verbose_feature_names_out = False,
    remainder = 'passthrough').set_output(transform = 'pandas') 



feature_eng_df = feature_eng.fit_transform(clean_ameS_x_train)
my_features = (feature_eng_df>> select (_.contains('_c_'))).columns.to_list()
 
preprocessor_1 = ColumnTransformer(
    transformers = [
        ('scaler', StandardScaler(), num_cols),
        ('just_select', 'passthrough', my_features ),
        ('OHE', OneHotEncoder(drop='first',handle_unknown='ignore' , sparse_output=False, min_frequency=16), make_column_selector(dtype_include  = 'object')),
    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough'  
    ).set_output(transform = 'pandas') 


prep_df = preprocessor_1.fit_transform(feature_eng_df)

interaction_transformer = PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)


drop_cols=['Alley_Pave', '3Ssn_Porch',
        	'Bldg_Type_2fmCon', 'House_Style_1Story',
        	'BsmtFin_Type_1_BLQ',
        	'BsmtFin_Type_1_Unf',
        	'BsmtFin_Type_2_BLQ',
        	'BsmtFin_Type_2_BLQ',
        	'BsmtFin_Type_2_Rec',
        	'Bsmt_Cond_infrequent_sklearn',
        	'Bsmt_Exposure_Mn',
        	'Bsmt_Full_Bath',
        	'Bsmt_Half_Bath',
        	'Condition_1_Feedr',
        	'Condition_2_infrequent_sklearn',
        	'Electrical_FuseF',
        	'Electrical_FuseF',
        	'Electrical_SBrkr',
        	'Electrical_infrequent_sklearn',
        	'Exter_Cond_Gd',
        	'Exter_Cond_TA',
        	'Exterior_1st_BrkFace',
        	'Exterior_1st_VinylSd',
        	'Exterior_2nd_MetalSd',
        	'Exterior_2nd_Plywood',
        	'Exterior_2nd_VinylSd','Exterior_2nd_Wd Shng',
        	'Exterior_2nd_Wd Sdng',
        	'Fence_infrequent_sklearn',
        	'Fireplace_Qu_infrequent_sklearn',
        	'Foundation_CBlock',
        	'Foundation_CBlock',
        	'Foundation_PConc',
        	# 'Foundation_Slab',
        	'Foundation_infrequent_sklearn',
        	'Functional_Min2',
        	'Functional_infrequent_sklearn',
        	'Garage_Cars',
        	'Garage_Cond_infrequent_sklearn',
        	'Garage_Finish_No_Garage',
        	'Garage_Qual_No_Garage_x_Garage_Prop_overGLA_c_',
        	'Garage_Type_BuiltIn',
        	'Garage_Type_Detchd',
        	'Garage_Type_infrequent_sklearn', 'Bsmt_Exposure_No',  
        	'Garage_Type_infrequent_sklearn',
        	'Heating_QC_Gd',
        	'House_Style_SFoyer','Bsmt_Qual_Gd',
        	'House_Style_SLvl',
        	'House_Style_infrequent_sklearn', 'Exterior_1st_infrequent_sklearn',
        	'Kitchen_Qual_infrequent_sklearn_x_Kitchen_AbvGr',
        	'Kitchen_Qual_infrequent_sklearn','Screen_Porch',
        	# 'Land_Contour_HLS',
        	# 'Land_Contour_Low', 'Land_Slope_Mod',
        	'Heating_infrequent_sklearn', ''
        	'Lot_Config_CulDSac', 'House_Style_2Story',
        	'Lot_Config_infrequent_sklearn',
        	'Lot_Shape_infrequent_sklearn', 'Lot_Shape_Reg',
        	'Lot_Shape_IR2', 'Lot_Config_Inside',
        	'Lot_Frontage', 'Heating_QC_Fa',
        	'Mas_Vnr_Area',#'TotRms_AbvGrd',
        	'Mas_Vnr_Type_None', 'Open_Porch_SF',
        	'Misc_Val','Area_p_car_c_', 'Misc_Feature_Shed',
        	'MS_SubClass',
        	'Neighborhood_NoRidge', 'MS_Zoning_infrequent_sklearn',
        	'Neighborhood_infrequent_sklearn','Overall_Cond_infrequent_sklearn',
        	# 'Overall_Qual_5','MS_Zoning_infrequent_sklearn',
        	# 'Overall_Qual_5_x_Year_Built',
        	# 'Overall_Qual_6',
        	# 'Overall_Qual_6_x_Year_Built',
        	'Street_infrequent_sklearn',
        	'Utilities_infrequent_sklearn']

interactions = ColumnTransformer(
  [('interaction_1', FunctionTransformer(
                                 cross_interactions  ,
                                 feature_names_out = None,
                                 kw_args={'num': 'Garage_Prop_overGLA_c_', 'cat': 'Garage_Qual', 'return_inputs':'both'}
                                 ), make_column_selector ('Garage_Qual|Garage_Prop_overGLA_c_')),
  ('inter_kitchen', FunctionTransformer(
                                 cross_interactions  ,
                                 feature_names_out = None,
                                 kw_args={'num': 'Kitchen_AbvGr', 'cat': 'Kitchen_Qual', 'return_inputs':'both'}
                                 ), make_column_selector ('Kitchen_Qual|Kitchen_AbvGr')),
  ('inter_central_y', FunctionTransformer(
                                 cross_interactions  ,
                                 feature_names_out = None,
                                 kw_args={'num': 'Gr_Liv_A/Loot_A_c_', 'cat': 'Area_Per_Room_c_', 'return_inputs':'both'}
                                 ), make_column_selector ('Area_Per_Room_c_|Gr_Liv_A/Loot_A_c_')),
  ('interaction_rooms_Bldg_Type', FunctionTransformer(
                                 cross_interactions  ,
                                 feature_names_out = None,
                                 kw_args={'num': '1s_Floor_prop_c_', 'cat': 'Overall_Q', 'return_inputs':'both'}
                                 ), make_column_selector ('1s_Floor_prop_c_|Overall_Q')),
  ('interaction_2', FunctionTransformer(
                                 cross_interactions  ,
                                 feature_names_out = None,
                                 kw_args={'num': 'Bsmt_Prop_c_', 'cat': 'Bsmt_Qua', 'return_inputs':'both'}
                                 ), make_column_selector ('Bsmt_Qual|Bsmt_Prop_c_'))
  ],
  verbose_feature_names_out = False,
  remainder='passthrough').set_output(transform = 'pandas')


step_rm =ColumnTransformer(
  [('TO_DROP', 'drop', drop_cols ),
  ('TO_DROP_BSMT', 'drop', make_column_selector('No_Basement|No_Garage|Exterior|Land|Fence') )],
  verbose_feature_names_out = False,
  remainder='passthrough').set_output(transform = 'pandas')

transformed_df = step_rm.fit_transform( interactions.fit_transform(prep_df))

pipeline = Pipeline([
  ('clean_data', clean_and_imputers),
  ('feat_eng', feature_eng),
  ('preprocessor', preprocessor_1),
  ('interactions', interactions),
  ('remove', step_rm),
  ('regressor', LinearRegression())])

# Entrenar el pipeline
pipeline.fit(ames_x_train, ames_y_train)  
##### Extracción de coeficientes
coefs = pd.DataFrame({'variable':transformed_df.columns.to_list(),
              'coefs': pipeline.named_steps['regressor'].coef_.round(5)})


model = sm.OLS(ames_y_train, sm.add_constant(transformed_df) ).fit()


 
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
    >>left_join(_, coefs, on = 'variable')
)


p_vals >> top_n(20,_.p_val)
model.summary2()

## PREDICCIONES
y_pred = pipeline.predict(ames_x_test) *ames_x_test.Gr_Liv_Area
y_obs = Sale_Price_test


##### Métricas de desempeño


predictores = transformed_df.columns.to_list()
get_metrics(y_pred, y_obs,predictors= len(predictores) )

# 
# y_pred = pipeline.predict(ames_x_val)*ames_x_val.Gr_Liv_Area
# 
# get_metrics(y_pred,Sale_Price_validation,len(transformed_df.columns ) )


