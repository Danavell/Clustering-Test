import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def Add_Means(series, columns):
    for column in columns:
        series[column].replace(
              np.nan, 
              Calculate_Means(series, column), 
              inplace=True)

def Calculate_Means(series, column):
    return series[column].astype('f').mean(axis=0)




headers = ["symboling","normalized-losses","make","fuel-type","aspiration", 
           "num-of-doors","body-style", "drive-wheels","engine-location",
           "wheel-base", "length","width","height","curb-weight","engine-type",
           "num-of-cylinders", "engine-size","fuel-system","bore","stroke",
           "compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg",
           "price"]

car_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data', 
                     na_values = '?', 
                     names=headers)




Add_Means(car_df, ['normalized-losses', 'bore', 'stroke','horsepower', 'peak-rpm'])
        
car_df['price'].astype('f').mean(axis=0)

car_df.dropna(subset=['price'], inplace=True)
car_df['num-of-doors'].replace(np.nan, 'four', inplace=True)

car_df['city-L/100km'] = 235/car_df['city-mpg']
car_df['highway-L/100km'] = 235/car_df['highway-mpg']

car_df["horsepower"] = car_df["horsepower"].astype(float, copy=True)
binwidth = (max(car_df["horsepower"])-min(car_df["horsepower"]))/4
bins = np.arange(min(car_df["horsepower"]), max(car_df["horsepower"]), binwidth)
group_names = ['Low', 'Medium', 'High']
car_df['horsepower-binned'] = pd.cut(car_df['horsepower'], bins, labels=group_names,include_lowest=True )

cond = car_df < 5

small_df = car_df.loc[:, ['make', 'body-style', 'price']]

car_df.to_excel('auto_data.xlsx', index=False)
car_df.to_csv('auto_data.txt', index=False)

writer = pd.ExcelWriter('Multiple.xlsx', engine='xlsxwriter')
small_df.to_excel(writer, sheet_name='Preview', index=False)
car_df.to_excel(writer, sheet_name='Full', index=False)
writer.save()

grouped = car_df.groupby(['make', 'body-style'])
grouped = grouped['price'].agg([('Average', 'mean')])
grouped['Average_Price'] = car_df['price'].groupby(car_df['make']).mean()

missing_data = car_df.isnull()
data = dict()
for column in missing_data.columns.values.tolist():
    data[column] = missing_data[column].value_counts()
data = pd.DataFrame(data)
data = data.T

desc = car_df.describe()