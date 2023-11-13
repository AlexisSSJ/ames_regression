from sklearn.compose import make_column_selector

ames_x_train.Lot_Frontage.mean()

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

clean_ameS_x_train= clean_and_imputers.fit_transform(ames_x_train)
