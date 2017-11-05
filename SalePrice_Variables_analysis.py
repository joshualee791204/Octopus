import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
from Individual_Variables_Analysis import summary_indetail as summ
##load train data
raw_train = pd.read_csv('train.csv')
df=raw_train.copy()

#This funtion plot the relationshipn between y and other variables. Float, int, or str use different types of plots
def plot_saleprice_vars(df,range_index):
    plt.close('all')
    try:
        for i in range_index:
            if type(df.iloc[:,i].dropna().values[0]) is np.float64:
                data=pd.concat([df.SalePrice, df.iloc[:,i]], axis=1)
                plt.figure(i)
                data.plot.scatter(x=df.columns[i], y="SalePrice", ylim=(0,800000))
                plt.title("Number of NAs is %i" % (1460-summ(df).loc[summ(df).name==df.columns[i]].num_not_NAs))
            else:
                if type(df.iloc[:,i].dropna().values[0]) is np.int64:
                    data=pd.concat([df.SalePrice, df.iloc[:,i]], axis=1)
                    plt.figure(i)
                    data.plot.scatter(x=df.columns[i], y="SalePrice", ylim=(0,800000))
                    plt.title("Number of NAs is %i" % (1460-summ(df).loc[summ(df).name==df.columns[i]].num_not_NAs))
                else:
                    data=pd.concat([df.SalePrice, df.iloc[:,i]], axis=1)
                    plt.figure(i)
                    sns.boxplot(x=df.columns[i], y='SalePrice', data=data)
                    plt.title("Number of NAs is %i" % (1460-summ(df).loc[summ(df).name==df.columns[i]].num_not_NAs))
        plt.show()
    except:
        print("Error: index out of range")
