import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
from Individual_Variables_Analysis import summary_indetail as summ
##load train data
raw_train = pd.read_csv('train.csv')
df=raw_train.copy()

#this function will get the column name and return the number of NAs, num of duplicates excluding NAs.
def summary_indetail(df):
    summ = pd.DataFrame(df.columns, columns=['name'])
    summ['type'] = [type(df[col][df[col].first_valid_index()]) for col in df.columns]
    summ['num_not_NAs'] = [df[col].dropna().shape[0] for col in df.columns]
    summ['num_dupl'] = [df[col].dropna().nunique() for col in df.columns]
    summ['ratio_NAs'] = [(1460-x)/1460 for x in summ['num_not_NAs']]
    return summ

#This funtion plot the relationshipn between SalePrice and other variables. Float, int, or str use different types of plots
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


#This function will plot the columns with df and range of column index as input.
def plot_individual_columns(df,range_col):
    try:
        for i in range_col:
            serie = df.iloc[:,i]
            ##There are three classes for all the columns: float, integer, and str
            if type(df.iloc[:,i].dropna().values[0]) is np.float64:
                plt.figure(i)
                sns.distplot(df.iloc[:,i].dropna(), label=('%s' %(df.columns[i])))
            else:
                if type(df.iloc[:,i].dropna().values[0]) is np.int64:
                    plt.figure(i)
                    plt.hist(df.iloc[:,i])
                    plt.xlabel('%s' %(df.columns[i]))
                else:
                    ##The category columns are not plotted. 
                    pass
            plt.show()
    except IndexError:
        print("Error: The index selected is out of the range")

 

