testing_data = pd.read_csv('data/proyecto_ames_test.csv').rename(columns = lambda x: x.replace (' ','_'))

preds = (
    testing_data 
        >> mutate (Sale_Price_pred = pipeline_linreg.predict(testing_data) * _.Gr_Liv_Area)
        >> select (_.Sale_Price_pred)
)

preds.to_csv('data/predictions.csv')
