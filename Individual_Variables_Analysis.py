import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib

raw_df = pd.read_csv('train.csv')
df = raw_df.copy()

#this function will get the column name and return the number of NAs, num of duplicates excluding NAs.
def summary_indetail(df):
    summ = pd.DataFrame(df.columns, columns=['name'])
    summ['type'] = [type(df[col][df[col].first_valid_index()]) for col in df.columns]
    summ['num_not_NAs'] = [df[col].dropna().shape[0] for col in df.columns]
    summ['num_dupl'] = [df[col].dropna().nunique() for col in df.columns]
    summ['ratio_NAs'] = [(1460-x)/1460 for x in summ['num_not_NAs']]
    return summ

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

 
