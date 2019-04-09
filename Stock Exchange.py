import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc

StockExchangeData = {
    'Year': [2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2016, 2016, 2016, 2016, 2016, 2016,
             2016, 2016, 2016, 2016, 2016, 2016],
    'Month': [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'Interest_Rate': [2.75, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.25, 2.25, 2.25, 2, 2, 2, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
                      1.75, 1.75, 1.75, 1.75, 1.75],
    'Unemployment_Rate': [5.3, 5.3, 5.3, 5.3, 5.4, 5.6, 5.5, 5.5, 5.5, 5.6, 5.7, 5.9, 6, 5.9, 5.8, 6.1, 6.2, 6.1, 6.1,
                          6.1, 5.9, 6.2, 6.2, 6.1],
    'Stock_Index_Price': [1464, 1394, 1357, 1293, 1256, 1254, 1234, 1195, 1159, 1167, 1130, 1075, 1047, 965, 943, 958,
                          971, 949, 884, 866, 876, 822, 704, 719]
    }

df = pd.DataFrame(StockExchangeData,
                  columns=['Year', 'Month', 'Interest_Rate', 'Unemployment_Rate', 'Stock_Index_Price'])

sub = df[['Interest_Rate', 'Unemployment_Rate']]
print(sub[:-1])
Xtrain, Xtest, ytrain, ytest = train_test_split(sub, df['Stock_Index_Price'], test_size=0.9)
lin = LinearRegression().fit(Xtrain, ytrain)

y_pred = lin.predict(Xtest)
print("R^2: {}".format(lin.score(Xtest, ytest)))
rmse = np.sqrt(mean_squared_error(ytest.astype('float'), y_pred))
print("Root Mean Squared Error: {}".format(rmse))

sub = df[['Interest_Rate', 'Unemployment_Rate', 'Stock_Index_Price']]


def ecdf(s):
    x = np.sort(s)
    y = np.arange(0, len(s) + 1) / len(s)
    return x, y


def plot_ecdfs(data):
    for key in data.keys():
        x, y = ecdf(data[key])
        print(x, y)
        break


plot_ecdfs(sub)

df.info()
df.describe
print(df.columns)

grouped = sub.groupby(['Interest_Rate', 'Unemployment_Rate'], as_index=False).mean()
grouped_pivot = sub.pivot(index='Interest_Rate', columns='Unemployment_Rate')
grouped_pivot

ax = sns.heatmap(grouped, cmap="YlGnBu")
_ = plt.plot(df['Interest_Rate'], df['Unemployment_Rate'])

plt.show()import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc


StockExchangeData = {'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]
                }

df = pd.DataFrame(StockExchangeData, columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])


sub = df[['Interest_Rate','Unemployment_Rate']]
print(sub[:-1])
Xtrain, Xtest, ytrain, ytest = train_test_split(sub, df['Stock_Index_Price'], test_size=0.9)
lin = LinearRegression().fit(Xtrain, ytrain)

y_pred = lin.predict(Xtest)
print("R^2: {}".format(lin.score(Xtest, ytest)))
rmse = np.sqrt(mean_squared_error(ytest.astype('float'), y_pred))
print("Root Mean Squared Error: {}".format(rmse))




sub = df[['Interest_Rate','Unemployment_Rate', 'Stock_Index_Price']]

def ecdf(s):
    x = np.sort(s)
    y = np.arange(0, len(s) + 1) / len(s)
    return x, y

def plot_ecdfs(data):
    for key in data.keys():
        x, y = ecdf(data[key])
        print(x, y)
        break

plot_ecdfs(sub)


df.info()
df.describe
print(df.columns)

grouped = sub.groupby(['Interest_Rate','Unemployment_Rate'], as_index= False).mean()
grouped_pivot = sub.pivot(index='Interest_Rate', columns='Unemployment_Rate')
grouped_pivot


ax = sns.heatmap(grouped, cmap="YlGnBu")
_ = plt.plot(df['Interest_Rate'], df['Unemployment_Rate'])



plt.show()