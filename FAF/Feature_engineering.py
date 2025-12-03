def feature_engineering(Big_table_name : str, New_big_table_name :str):
    import pandas as pd
    df = pd.read_csv(f"{Big_table_name}.csv")

    '''Creating Average Speed'''
    df['Average_Speed'] = df['distance_m'] / df['duration_s']

    '''Adding Turn Density'''
    df['Turn_Density'] = df['turns'] / (df['distance_m'] / 1000)

    '''Saving new DF as a file'''
    df.to_csv(f'{New_big_table_name}.csv')

    '''return dataframe'''
    return (df.head(10))


if __name__ == "__main__":
    feature_engineering (Big_table_name = , New_big_table_name = )
