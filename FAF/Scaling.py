def scaling():
    import pandas as pd
    df = pd.read_csv("UK_Engineered_Data.csv")
    pd.set_option('display.max_columns', None)

    X = df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'id', 'name'], axis=1)
#Scalling function will be done later when we are happy with the model 
