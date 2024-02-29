from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder,FunctionTransformer
from sklearn.compose import ColumnTransformer, 
from sklearn.impute import SimpleImputer
ames_x_train.Lot_Frontage.mean()
from sklearn.pipeline import Pipeline


#Sólo 4 casas tienen Piscina/ por lo tanto esta variable no es útil
ames_x_train >> select(_.contains('Pool')) >> filter (_.Pool_Area==0)


n_usefull = ['Sale_Type', 'Sale_Condition', 'Pool_QC', 'Pool_Area', 'Yr_Sold']

# NA de Alley are no_alley ACCES
# na de Mas_Vnr_Type es none
# NA EN Bsmt_Qual es No_basement
#remember quit "No_Basement" cat
# electrical imput mode
# no fire place imput 'No_Fireplace'

clean_and_imputers = ColumnTransformer(
    [('not_usefull_variables', 'drop', n_usefull),
     ('Impute_Alley', SimpleImputer(
         missing_values=np.nan,
         strategy='constant', 
         fill_value= 'No_Alley_Access'), ['Alley']),
     ('Impute_Mas_Vnr_Type', SimpleImputer(
         missing_values=np.nan,
         strategy='constant', 
         fill_value= 'None'), ['Mas_Vnr_Type']),
     ('Impute_Mas_Vnr_Area', SimpleImputer(
         missing_values=np.nan,
         strategy='constant', 
         fill_value= 0), ['Mas_Vnr_Area']),
     ('Impute_Lot_Frontage', SimpleImputer(
         missing_values=np.nan,
         strategy='mean'), ['Lot_Frontage']),
     ('Impute_Fireplace_Qu', SimpleImputer(
         missing_values=np.nan,
         strategy='constant', 
         fill_value= 'No_Fireplace'), ['Fireplace_Qu']),
     ('Impute_Electrical', SimpleImputer(
         missing_values=np.nan,
         strategy='most_frequent'),['Electrical']),
     ('Impute_Bsmt_Cat', SimpleImputer(
         missing_values=np.nan,
         strategy='constant', 
         fill_value= 'No_Basement'), make_column_selector(pattern = 'Bsmt',dtype_include = 'object')),
     ('Impute_Bsmt_Numeric', SimpleImputer(
         missing_values=np.nan,
         strategy='constant', 
         fill_value= 0), make_column_selector(pattern = 'Bsmt',dtype_exclude = 'object')),
     ('Impute_Garage', SimpleImputer(
         missing_values=np.nan,
         strategy='constant', 
         fill_value= 'No_Garage'), make_column_selector(pattern = 'Garage',dtype_include = 'object')),
     ('Impute_Garage_numeric', SimpleImputer(
         missing_values=np.nan,
         strategy='constant', 
         fill_value= 0), make_column_selector(pattern = 'Garage',dtype_exclude = 'object')),
     ('Impute_Fence', SimpleImputer(
         missing_values=np.nan,
         strategy='constant', 
         fill_value= 'No_Fence'), ['Fence']),
     ('Impute_Misc_Feature', SimpleImputer(
         missing_values=np.nan,
         strategy='constant', 
         fill_value= 'None'), ['Misc_Feature'])
    ],
    remainder= 'passthrough',
    verbose_feature_names_out = False
).set_output(transform = 'pandas')

clean_ames_x_train= clean_and_imputers.fit_transform(ames_x_train)


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


feature_eng_df = feature_eng.fit_transform(clean_ames_x_train)
my_features = (feature_eng_df>> select (_.contains('_c_'))).columns.to_list()

preprocessor_1 = ColumnTransformer(
    transformers = [
        ('scaler', StandardScaler(), make_column_selector(dtype_exclude  = 'object')),
        # ('just_select', 'passthrough', my_features ),
        ('OHE', OneHotEncoder(drop='first',handle_unknown='infrequent_if_exist' , sparse_output=False, min_frequency = 12), make_column_selector(dtype_include  = 'object'))],
    verbose_feature_names_out = False,
    remainder = 'drop').set_output(transform = 'pandas')

preprocesed = preprocessor_1.fit_transform(feature_eng_df)


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
  ('Gr_Liv_Area_x_Neighborhood', FunctionTransformer(
                                 cross_interactions, feature_names_out = None,
                                 kw_args={'num': 'Gr_Liv_Area', 'cat': 'Neighborhood', 'return_inputs':'both'}
                                 ), make_column_selector ('Gr_Liv_Area|Neighborhood')),
  ('TO_DROP_', 'drop', make_column_selector('No_Basement|No_Garage|Exterior|Land|Fence')) ],
  verbose_feature_names_out = False,
  remainder='passthrough').set_output(transform = 'pandas')
  
 interactions.fit_transform(preprocesed)
  
 get_df = Pipeline([ 
  ('clean_data', clean_and_imputers),
  ('feat_eng', feature_eng),
  ('preprocessor', preprocessor_1),
  ('interactions', interactions)])
  
get_df.fit_transform(ames_x_train)
