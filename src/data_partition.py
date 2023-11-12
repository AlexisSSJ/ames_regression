from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression, mutual_info_regression
from plydata.one_table_verbs import pull
from plotnine import *

import numpy as np
from plydata.tidy import pivot_wider, pivot_longer
from mizani.formatters import comma_format, dollar_format
import pandas as pd
from siuba import *



#Crear transformación de la variable de respuesta
ames = pd.read_csv("data/ames.csv") >> mutate (Price_4_GLA= _.Sale_Price/_.Gr_Liv_Area)


ames.info()


y = ames >>  pull("Price_4_GLA")
X = select(ames, -_.Sale_Price, -_.Price_4_GLA)

X=(X >> mutate(
                wood_prop= _.Wood_Deck_SF/_.Gr_Liv_Area,
                area_per_car=case_when({
                              _.Garage_Cars == 0:0,
                              True:_.Garage_Area/ _.Garage_Cars}),
                last_remod = case_when({
                              _.Year_Built>_.Year_Remod_Add: 2023-_.Year_Remod_Add,
                              True: 2023-_.Year_Built}),
                Mo_Sold= _.Mo_Sold.astype('object')
                )
                
            )

numeric_column = ames >> pull("Price_4_GLA")
quartiles = np.percentile(numeric_column, [25, 50, 75])

# Crea una nueva variable categórica basada en los cuartiles
stratify_variable = pd.cut(
 numeric_column, 
 bins=[float('-inf'), quartiles[0], quartiles[1], quartiles[2], float('inf')],
 labels=["Q1", "Q2", "Q3", "Q4"]
 )

ames_x_train, ames_x, ames_y_train, ames_y = train_test_split(
                       X, y, 
                       test_size = 0.30, 
                       random_state = 12345, 
                       stratify = stratify_variable
                       )

stratify_variable = pd.cut(
 ames_y , 
 bins=[float('-inf'), quartiles[0], quartiles[1], quartiles[2], float('inf')],
 labels=["Q1", "Q2", "Q3", "Q4"]
 )

ames_x_test, ames_x_val, ames_y_test, ames_y_val = train_test_split(
       ames_x, ames_y, 
       test_size = 1/3, 
       random_state = 12345, 
       stratify = stratify_variable
       )
       
 = ames_y_train * ames_x_train.Gr_Liv_Area
Sale_Price_test = ames_y_test * ames_x_test.Gr_Liv_Area
Sale_Price_validation = ames_y_val * ames_x_val.Gr_Liv_Area
print('Totales:\t', ames.shape[0],'\nTraining:\t', ames_y_train.shape[0],'\nTesting:\t', ames_y_test.shape[0],'\nValidation:\t', ames_y_val.shape[0])




