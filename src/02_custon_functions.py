### Fuciones personalizadas


###### IMPORTTANCIA DE VARIABLES
#     PARAMETROS:
##  test_fram:          dataframe de las features para testing
##  y_obs:              labes de los datos de testing 
##  seleccted_columns:  Variables que contempla el modelo\
##  actual_mse:         MSE calculado en test con todas las variables
##  pipeline:           pipeline para hacer las predcciones
##  trans_pred:         Poner True si las predicciones se están haciendo por 
## n_permutations=50
def importance_from_model (test_frame, y_obs, selected_columns, pipeline, actual_mse, n_permutations=50 ,trans_pred= False ):
  performance_losses = []
  
  for i in range(test_frame[selected_columns].shape[1]):
    loss = []
    for j in range(n_permutations):
        test_frame_permuted = test_frame[selected_columns].copy()
        test_frame_permuted.iloc[:, i] = np.random.permutation(test_frame_permuted.iloc[:, i])
        if trans_pred == False:
            y_pred_permuted = pipeline.predict(test_frame_permuted)
        else:
            y_pred_permuted = pipeline.predict(test_frame_permuted)* test_frame_permuted.Gr_Liv_Area
        mse_permuted = mean_squared_error(y_obs, y_pred_permuted)
        loss.append(mse_permuted)
    performance_losses.append(loss)
  
  performance_losses = performance_losses/np.sum(performance_losses, axis=0)
  mean_losses = np.mean(performance_losses, axis=1)
  std_losses = np.std(performance_losses, axis=1)
  
  importance_df = pd.DataFrame({
  'Variable': selected_columns, 
  'Mean_Loss': mean_losses, 
  'Std_Loss': std_losses
  })
  return importance_df
  

def adjusted_r2_score(y_true, y_pred, n, p):
  r2 = r2_score(y_true, y_pred)
  adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
  return adjusted_r2




def div_columns(X, c1, c2):
    X["c1_c2"] = X[c1]/ X[c2]
    return X



def adj_r(y_o, y_p , p, n):
  r2_adj = 1 - (n - 1) / (n - p - 1) * (1 - r2_score(y_o, y_p))
  
  
def get_metrics (y_pred, y_obs,predictors):
  me = np.mean(y_obs - y_pred)
  mae = mean_absolute_error(y_obs, y_pred)
  mape = mean_absolute_percentage_error(y_obs, y_pred)
  mse = mean_squared_error(y_obs, y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_obs, y_pred)

  n = len(y_obs)  # Número de observaciones
  p = predictors  # Número de predictores 
  r2_adj = 1 - (n - 1) / (n - p - 1) * (1 - r2)
  

  metrics_data = {
      "Metric": ["ME", "MAE", "MAPE", "MSE", "RMSE", "R2", "R2Adj"],
      "Value": [me, mae, mape, mse, rmse, r2, r2_adj]
  }
  return pd.DataFrame(metrics_data).set_index('Metric')



def rmv_elements (list_of_elements, complete_list):
  for element in list_of_elements:
    complete_list.remove(element)
  return complete_list



def validation_results(x_val,y_val,  pipeline , n_preds):
  y_pred = pipeline.predict(x_val)
  
  ames_test = (
    ames_x_val >>
    mutate(Sale_Price_Pred = y_pred, Sale_Price =y_val))
  
  ##### Métricas de desempeño
  
  y_obs = ames_test["Sale_Price"]
  y_pred = ames_test["Sale_Price_Pred"]
  return get_metrics(y_pred, y_obs, n_preds)


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def check_cv (df, param_p = False):
  if param_p == False:
    (df 
      >> mutate(RMSE = abs(_.mean_test_neg_mean_squared_error)**0.5, MAPE = abs (_.mean_test_mape))
      >> pivot_longer(
                      cols=['mean_test_r2', 'MAPE','RMSE'],
                      names_to='parameter', 
                      values_to='value')
      >> ggplot( aes(x = "param_n_neighbors", y = "value", shape = "param_weights", color= 'param_metric')) 
      + geom_point(alpha = 0.9,position=position_dodge(width=0.1))
      + facet_wrap("~parameter",ncol =1, scales = "free_y")
      + labs( y = '',x= 'Parámetro: vecinos cercanos K' ,shape = 'Ponderación',color = 'Métrica' )).draw(True)
  else:
    (df 
      >> mutate(RMSE = abs(_.mean_test_neg_mean_squared_error)**0.5 , MAPE = abs (_.mean_test_mape))
      >> pivot_longer(
                      cols=['mean_test_r2', 'MAPE','RMSE'],
                      names_to='parameter', 
                      values_to='value')
      >> ggplot( aes(x = "param_n_neighbors", y = "value", shape = "param_weights", color= 'param_metric',size = 'param_p')) 
      + geom_point(alpha = 0.9,position=position_dodge(width=0.1))
      + facet_wrap("~parameter",ncol =1, scales = "free_y")
      + labs( y = '',x= 'Parámetro: vecinos cercanos K' ,shape = 'Ponderación',color = 'Métrica' )).draw(True)



