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



ames_x_test
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

def cross_interactions2 (X, num, cat, return_inputs = 'none'):
    return_cols = []
    df = X.filter(regex=(cat)).copy()
    df2 = X.filter(regex=(num)).copy()
    for col_num in df2.columns.to_list():
      for column in df.columns.to_list():
        name = column + '_x_'+ col_num
        X[name] = df[column] * X[col_num]
        return_cols.append(name)
    if return_inputs == 'none':
      return X[return_cols]
    elif return_inputs == 'both':
      return X[return_cols + df.columns.to_list() + df2.columns.to_list()]
    elif return_inputs == 'num':
      return X[return_cols + df2.columns.to_list()]
    else:
      return X[return_cols+ df.columns.to_list()]

def antique_func(X,yb, yr):
  return (X >> mutate (Antique_c_ = 2023 - _[yb], Last_remod_c_ = 2023 - _[yr], Antique2_c_=(2023 - _[yb])**2)
          >> select (_.contains ('_c_')))



cond ={1: 'Poor', 2:'Poor',3:'Poor',4:'Average', 5:'Average',6:'Average', 7:'Good_or_Excellent', 8:'Good_or_Excellent', 9:'Good_or_Excellent', 10 :'Good_or_Excellent'}

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
    ('MVNRA_prop',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Mas_Vnr_Area', 'c2': 'Lot_Area', 'feature_name' : 'MVNRA_propLA'}
                                 ), ['Mas_Vnr_Area', 'Lot_Area']),
    ('2nd_Flr_SF_propGLA',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': '2nd_Flr_SF', 'c2': 'Gr_Liv_Area', 'feature_name' : '2nd_Flr_SF_propGLA'}
                                 ), ['2nd_Flr_SF', 'Gr_Liv_Area']),
    ('Bsmt_Prop',  FunctionTransformer(
                                 div_columns,
                                 feature_names_out = None,
                                 kw_args={'c1': 'Total_Bsmt_SF', 'c2': 'Gr_Liv_Area', 'feature_name' : 'Bsmt_Prop'}
                                 ),['Total_Bsmt_SF', 'Gr_Liv_Area'] ),
    ('Time_var',  FunctionTransformer(
                                 antique_func,
                                 feature_names_out = None,
                                 kw_args={'yb': 'Year_Built', 'yr': 'Year_Remod/Add'}
                                 ), ['Year_Built', 'Year_Remod/Add']),
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
    ('just_select', 'passthrough', ['Total_Bsmt_SF','2nd_Flr_SF', 'TotRms_AbvGrd', '1st_Flr_SF', 'Year_Built', 'Year_Remod/Add', 'Garage_Area', 'Lot_Area', 'Gr_Liv_Area', 'Mas_Vnr_Area','Wood_Deck_SF' ])],
    verbose_feature_names_out = False,
    remainder = 'passthrough').set_output(transform = 'pandas')


feature_eng_df = feature_eng.fit_transform(clean_ameS_x_train)
my_features = (feature_eng_df>> select (_.contains('_c_'))).columns.to_list()

preprocessor_1 = ColumnTransformer(
    transformers = [
        ('scaler', StandardScaler(), num_cols),
        ('just_select', 'passthrough', my_features ),
        ('OHE', OneHotEncoder(drop='first',handle_unknown='infrequent_if_exist' , sparse_output=False, min_frequency = 12), make_column_selector(dtype_include  = 'object'))],
    verbose_feature_names_out = False,
    remainder = 'passthrough').set_output(transform = 'pandas')


prep_df = preprocessor_1.fit_transform(feature_eng_df)


interactions = ColumnTransformer(
  [('Garage_interaction', FunctionTransformer(
                                 cross_interactions, feature_names_out = None,
                                 kw_args={'num': 'Garage_Prop_overGLA_c_', 'cat': 'Garage_Qual', 'return_inputs':'both'}
                                 ), make_column_selector ('Garage_Qual|Garage_Prop_overGLA_c_')),
  ('inter_kitchen', FunctionTransformer(
                                 cross_interactions, feature_names_out = None,
                                 kw_args={'num': 'Kitchen_AbvGr', 'cat': 'Kitchen_Qual', 'return_inputs':'both'}
                                 ), make_column_selector ('Kitchen_Qual|Kitchen_AbvGr')),
  ('Last_remod_c_', FunctionTransformer(
                                 cross_interactions, feature_names_out = None,
                                 kw_args={'num': 'Gr_Liv_A/Loot_A_c_', 'cat': 'Gr_Liv_A/Loot_A_c_', 'return_inputs':'num'}
                                 ), make_column_selector ('Gr_Liv_A/Loot_A_c_')),
  ('1s_Floor_prop_c_|Overall_Q', FunctionTransformer(
                                 cross_interactions, feature_names_out = None,
                                 kw_args={'num': '1s_Floor_prop_c_', 'cat': 'Overall_Q', 'return_inputs':'both'}
                                 ), make_column_selector ('1s_Floor_prop_c_|Overall_Q')),
  ('Bsmt_Qual|Bsmt_Prop_c_', FunctionTransformer(
                                 cross_interactions, feature_names_out = None,
                                 kw_args={'num': 'Bsmt_Prop_c_', 'cat': 'Bsmt_Qual', 'return_inputs':'both'}
                                 ), make_column_selector ('Bsmt_Qual|Bsmt_Prop_c_')),
  ('Area_Per_Room_c_|Overall', FunctionTransformer(
                                 cross_interactions, feature_names_out = None,
                                 kw_args={'num': '2nd_Flr_SF_propGLA_c_', 'cat': 'Antique', 'return_inputs':'both'}
                                 ), make_column_selector ('2nd_Flr_SF_propGLA_c_|Antique')),
  ('MVNRA_propGLA_c_|Mas_Vnr_Type', FunctionTransformer(
                                 cross_interactions, feature_names_out = None,
                                 kw_args={'num': 'MVNRA_propLA_c_', 'cat': 'Mas_Vnr_Type', 'return_inputs':'both'}
                                 ), make_column_selector ('MVNRA_propLA_c_|Mas_Vnr_Type')),
  ('Wood_propGLA_c_|Fireplaces', FunctionTransformer(
                                 cross_interactions, feature_names_out = None,
                                 kw_args={'num': 'Wood_propGLA_c_', 'cat': 'Fireplaces', 'return_inputs':'both'}
                                 ), make_column_selector ('Wood_propGLA_c_|Fireplaces')),
  ('Wood_propGLA_c_|Antique', FunctionTransformer(
                                 cross_interactions, feature_names_out = None,
                                 kw_args={'num': 'Gr_Liv_Area', 'cat': 'Neighborhood', 'return_inputs':'both'}
                                 ), make_column_selector ('Gr_Liv_Area|Neighborhood_C|Neighborhood_[STN]')),
  ('TO_DROP_', 'drop', make_column_selector('No_Basement|No_Garage|Exterior|Land|Fence')) ],
  verbose_feature_names_out = False,
  remainder='passthrough').set_output(transform = 'pandas')






## drop cols
drop_cols=['2nd_Flr_SF',
		'3Ssn_Porch',
		'Neighborhood_IDOTRR',
		'Kitchen_Qual_Gd',
		'Alley_No_Alley_Access',
		'Neighborhood_StoneBr_x_Gr_Liv_Area',
		'BsmtFin_SF_1',
		'Alley_Pave',
		'Neighborhood_BrkSide',
		'Neighborhood_SawyerW_x_Gr_Liv_Area',
		# 'Neighborhood_SawyerW'
		'Neighborhood_Edwards',
		'Bedroom_AbvGr',
		'Kitchen_Qual_Fa',
		'Bldg_Type_2fmCon',
		'Neighborhood_Timber_x_Gr_Liv_Area',
		'Neighborhood_Timber',
		'Neighborhood_NWAmes_x_Gr_Liv_Area',
		'Neighborhood_Somerst_x_Gr_Liv_Area',
		'Bldg_Type_Duplex','Condition_1_infrequent_sklearn',
		# 'Neighborhood_MeadowV_x_Gr_Liv_Area',
		'Roof_Style_infrequent_sklearn',
		'Bldg_Type_Twnhs',
		'Bldg_Type_TwnhsE',
		'BsmtFin_SF_2',
		'BsmtFin_Type_1_BLQ',
		'BsmtFin_Type_1_GLQ',
		'Neighborhood_SWISU_x_Gr_Liv_Area',
		'Mas_Vnr_Type_None_x_MVNRA_propLA_c_',
		# 'Bsmt_Qual_Fa',
		'Neighborhood_ClearCr',
		'BsmtFin_Type_1_LwQ',
		'BsmtFin_Type_1_Rec',
		'Kitchen_Qual_TA',
		# 'TotRms_AbvGrd',
		'BsmtFin_Type_1_Unf',
		'BsmtFin_Type_2_BLQ',
		'BsmtFin_Type_2_GLQ',
		'BsmtFin_Type_2_LwQ','Neighborhood_CollgCr_x_Gr_Liv_Area',
		# '1st_Flr_SF',
		# 'Mas_Vnr_Type_None_x_MVNRA_propGLA_c_',
		# 'Mas_Vnr_Type_Stone_x_MVNRA_propGLA_c_',
		'BsmtFin_Type_2_Rec',
		# 'Neighborhood_Mitchel_x_Gr_Liv_Area',
		# 'Mas_Vnr_Type_infrequent_sklearn_x_MVNRA_propGLA_c_',
		'BsmtFin_Type_2_Unf',
		'Kitchen_Qual_TA_x_Kitchen_AbvGr',
		'Bsmt_Cond_Gd',
		'Fireplace_Qu_Po',
		'Bsmt_Cond_TA',
		'Bsmt_Cond_infrequent_sklearn',
		'Bsmt_Exposure_Mn',
		'Bsmt_Exposure_No',
		'Foundation_PConc',
		# 'Heating_QC_Fa',
		# 'Functional_infrequent_sklearn',
		# 'Mas_Vnr_Type_infrequent_sklearn',
		'Bsmt_Full_Bath',
		'Bsmt_Half_Bath',
		# 'Bsmt_Qual_Fa',
		# 'Condition_1_Norm',
		'Bsmt_Qual_Gd_x_Bsmt_Prop_c_',
		# 'Exter_Qual_TA',
		# 'Exter_Qual_Gd',
		# 'Exter_Qual_Fa',
		'Bsmt_Qual_No_Basement',
		'Bsmt_Qual_No_Basement_x_Bsmt_Prop_c_',
		'Mas_Vnr_Type_infrequent_sklearn_x_MVNRA_propLA_c_',
		'Bsmt_Qual_TA_x_Bsmt_Prop_c_',
		'Bsmt_Unf_SF',
		# 'Neighborhood_MeadowV',
		'Overall_Cond_Good_or_Excellent',
		'Central_Air_Y',
		'Condition_1_Feedr',
		'Condition_2_infrequent_sklearn',
		'Overall_Cond_Poor',
		# 'Fireplaces'
		# '1st_Flr_SF',
		# 'Functional_Typ',
		'Electrical_FuseF',
		'Electrical_SBrkr',
		'Electrical_infrequent_sklearn',
		# 'Mas_Vnr_Type_infrequent_sklearn',
		# 'Mas_Vnr_Type_None',
		'Enclosed_Porch',
		'Exter_Cond_Gd',
		'Exter_Cond_TA',
		'Fireplace_Qu_infrequent_sklearn',
		'Foundation_CBlock',
		'Foundation_Slab',
		'Foundation_infrequent_sklearn',
		'Full_Bath',
		'Functional_Min2',
		'Garage_Cars',
		'Garage_Cond_TA',
		'Garage_Cond_infrequent_sklearn',
		'Garage_Finish_RFn',
		'Garage_Finish_Unf',
		'Garage_Qual_No_Garage',
		'Garage_Qual_No_Garage_x_Garage_Prop_overGLA_c_',
		'Garage_Qual_infrequent_sklearn',
		'Garage_Qual_infrequent_sklearn_x_Garage_Prop_overGLA_c_',
		'Garage_Type_BuiltIn',
		'Garage_Type_Detchd',
		'Garage_Type_infrequent_sklearn',
		'Garage_Yr_Blt',
		'Half_Bath',
		'Heating_QC_Gd',
		'Heating_QC_infrequent_sklearn',
		'Heating_infrequent_sklearn',
		'House_Style_1Story',
		'House_Style_2Story',
		'House_Style_SFoyer',
		'House_Style_SLvl',
		'House_Style_infrequent_sklearn',
		'Kitchen_Qual_infrequent_sklearn',
		'Kitchen_Qual_infrequent_sklearn_x_Kitchen_AbvGr',
		'Last_remod_c_',
		'Lot_Area',
		'Lot_Config_CulDSac',
		'Lot_Config_FR2',
		'Lot_Config_Inside',
		'Lot_Config_infrequent_sklearn',
		'Lot_Frontage',
		'Lot_Shape_IR2',
		'Lot_Shape_Reg',
		'Lot_Shape_infrequent_sklearn',
		'Low_Qual_Fin_SF',
		'MS_SubClass',
		'MS_Zoning_RL',
		'MS_Zoning_RM',
		'MS_Zoning_infrequent_sklearn',
		'Mas_Vnr_Area',
		'Misc_Feature_Shed',
		'Mo_Sold',
		'Neighborhood_Crawfor',
		'Neighborhood_NoRidge',
		'Neighborhood_NridgHt',
		'Neighborhood_Somerst',
		'Neighborhood_infrequent_sklearn',
		'Open_Porch_SF',
		'Overall_Qual_Poor_x_1s_Floor_prop_c_',
		'Paved_Drive_P',
		'Paved_Drive_Y',
		'Roof_Style_Hip',
		'Roof_Matl_infrequent_sklearn',
		# 'Functional_infrequent_sklearn',
		'Wood_Deck_SF',
		'Garage_Area','Kitchen_Qual_Gd_x_Kitchen_AbvGr',
		'Screen_Porch',
		'Street_infrequent_sklearn',
		'Total_Bsmt_SF',
		'Garage_Qual_TA',
		'Kitchen_Qual_Fa_x_Kitchen_AbvGr',
		'Utilities_infrequent_sklearn'
		#############removing sf
            ]

## end

step_rm =ColumnTransformer(
  [('TO_DROP', 'drop', drop_cols )],
  verbose_feature_names_out = False,
  remainder='passthrough').set_output(transform = 'pandas')



pipeline_linreg = Pipeline([
  ('clean_data', clean_and_imputers),
  ('feat_eng', feature_eng),
  ('preprocessor', preprocessor_1),
  ('interactions', interactions),
  ('remove', step_rm),
  ('regressor', LinearRegression())])

# Entrenar el pipeline
pipeline_linreg.fit(ames_x_train, ames_y_train)

transformed_df = step_rm.fit_transform(interactions.fit_transform(prep_df))
##### Extracción de coeficientes
coefs = pd.DataFrame({
  'variable':transformed_df.columns.to_list(),
  'coefs': pipeline_linreg.named_steps['regressor'].coef_.round(5)
  })


model = sm.OLS(ames_y_train, sm.add_constant(transformed_df) ).fit()



p_vals = (
        model.pvalues.reset_index()
        >> select( _.variable == _.index, _.p_val == -1)
        >> mutate (s = case_when({
              _.p_val<0.025:'***',
              _.p_val<0.05:'**',
              _.p_val<0.1:'*',
              True:''}))
        >> arrange (_.s,- _.p_val)
        >>left_join(_, coefs, on = 'variable')
        )
p_vals >> top_n(20,_.p_val) >> select (_.coefs, _.p_val,_.s,_.variable)


model.summary2()

## PREDICCIONES
y_pred = pipeline_linreg.predict(ames_x_test) * ames_x_test.Gr_Liv_Area
y_obs = Sale_Price_test

##### Métricas de desempeño

predictores = transformed_df.shape[1]
get_metrics(y_pred, y_obs, predictores )

test = pd.DataFrame() >> mutate (Sale_Price = y_obs, Sale_Price_Pred = y_pred)

(
    ggplot(aes(x = y_pred, y =y_obs)) +
    geom_point() +
    scale_y_continuous(labels = dollar_format(digits=0, big_mark=','), limits = [0, 900000] ) +
    scale_x_continuous(labels = dollar_format(digits=0, big_mark=','), limits = [0, 900000] ) +
    geom_abline(color = "red") +
    coord_equal() +
    labs(
      title = "Comparación entre predicción y observación",
      x = "Predicción",
      y = "Observación")+theme_tufte()
)


(
test >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(x = "error")) +
  geom_histogram(color = "white", fill = "black") +
  geom_vline(xintercept = 0, color = "red") +
  scale_x_continuous(labels=dollar_format(big_mark=',', digits=0)) + 
  ylab("Conteos de clase") + xlab("Errores") +
  ggtitle("Distribución de error")+theme_tufte()
)

(
test >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(sample = "error")) +
  geom_qq(alpha = 0.3) + stat_qq_line(color = "red") +
  scale_y_continuous(labels=dollar_format(big_mark=',', digits = 0)) + 
  xlab("Distribución normal") + ylab("Distribución de errores") +
  ggtitle("QQ-Plot") +theme_tufte()
)

validation = pd.DataFrame() >> mutate (Sale_Price = Sale_Price_validation, Sale_Price_Pred = pipeline.predict(ames_x_val)*ames_x_val.Gr_Liv_Area )



results_val = (ames_x_val >> mutate (Pred= pipeline.predict(ames_x_val), true= ames_y_val))


(
  validation >>
    ggplot(aes(x = "Sale_Price_Pred", y = "Sale_Price")) +
    geom_point() +
    scale_y_continuous(labels = dollar_format(digits=0, big_mark=',') ) +
    scale_x_continuous(labels = dollar_format(digits=0, big_mark=',')) +
    geom_abline(color = "red") +
    coord_equal() +
    labs(
      title = "Comparación entre predicción y observación",
      x = "Predicción",
      y = "Observación")+theme_tufte()
)


(
validation >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(x = "error")) +
  geom_histogram(color = "white", fill = "black") +
  geom_vline(xintercept = 0, color = "red") +
  scale_x_continuous(labels=dollar_format(big_mark=',', digits=0)) + 
  ylab("Conteos de clase") + xlab("Errores") +
  ggtitle("Distribución de error")+theme_tufte()
)

(
validation >>
  select(_.Sale_Price, _.Sale_Price_Pred) >>
  mutate(error = _.Sale_Price - _.Sale_Price_Pred) >>
  ggplot(aes(sample = "error")) +
  geom_qq(alpha = 0.3) + stat_qq_line(color = "red") +
  scale_y_continuous(labels=dollar_format(big_mark=',', digits = 0)) + 
  xlab("Distribución normal") + ylab("Distribución de errores") +
  ggtitle("QQ-Plot") +theme_tufte()
)
