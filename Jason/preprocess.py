# Idea borrowed from this script: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
import pandas as pd
import numpy as np
##This is zeyu' model for imput
def impute(all_data):

	# PoolQC : data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.
	all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

	# MiscFeature : data description says NA means "no misc feature"
	all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

	# Alley : data description says NA means "no alley access"
	all_data["Alley"] = all_data["Alley"].fillna("None")

	# Fence : data description says NA means "no fence"
	all_data["Fence"] = all_data["Fence"].fillna("None")

	# FireplaceQu : data description says NA means "no fireplace"
	all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

	# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
	# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
	all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

	# GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
	for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
		all_data[col] = all_data[col].fillna('None')

	# GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
	for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'): # ??? Fine only if use as categorical feature
		all_data[col] = all_data[col].fillna(0)

	# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement
	for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
		all_data[col] = all_data[col].fillna(0)

	# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
	for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
		all_data[col] = all_data[col].fillna('None')

	# MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
	all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
	all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

	# MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'
	all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

	# Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
	all_data = all_data.drop(['Utilities'], axis=1)

	# Functional : data description says NA means typical
	all_data["Functional"] = all_data["Functional"].fillna("Typ")

	# Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
	all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

	# KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
	all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

	# Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
	all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
	all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

	# SaleType : Fill in again with most frequent which is "WD"
	all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

	# MSSubClass : Na most likely means No building class. We can replace missing values with None
	all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
	
	# Adding total sqfootage feature 
	all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

	# Transforming some numerical variables that are really categorical

	#MSSubClass=The building class
	all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

	#Changing OverallCond into a categorical variable
	all_data['OverallCond'] = all_data['OverallCond'].astype(str)

	#Year and month sold are transformed into categorical features.
	all_data['YrSold'] = all_data['YrSold'].astype(str)
	all_data['MoSold'] = all_data['MoSold'].astype(str)

	return all_data

##changed the YearBuilt, MszoneClass, MoSold, YrSold, YearBuilt, YearRemodAdd compared with basic
##changed pool_QC to binary
def impute_multi_1(all_data):
	# PoolQC : data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.
	
	all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

	# MiscFeature : data description says NA means "no misc feature"
	all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

	# Alley : data description says NA means "no alley access"
	all_data["Alley"] = all_data["Alley"].fillna("None")

	# Fence : data description says NA means "no fence"
	all_data["Fence"] = all_data["Fence"].fillna("None")

	# FireplaceQu : data description says NA means "no fireplace"
	all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

	# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
	# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
	all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

	# GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
	for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
		all_data[col] = all_data[col].fillna('None')

	# GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
	for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'): # ??? Fine only if use as categorical feature
		all_data[col] = all_data[col].fillna(0)

	# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement
	for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
		all_data[col] = all_data[col].fillna(0)

	# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
	for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
		all_data[col] = all_data[col].fillna('None')

	# MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
	all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
	all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

	# MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'
	all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

	# Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
	all_data = all_data.drop(['Utilities'], axis=1)

	# Functional : data description says NA means typical
	all_data["Functional"] = all_data["Functional"].fillna("Typ")

	# Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
	all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

	# KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
	all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

	# Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
	all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
	all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

	# SaleType : Fill in again with most frequent which is "WD"
	all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

	# MSSubClass : Na most likely means No building class. We can replace missing values with None
	all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
	
	# Adding total sqfootage feature, add boolean feature for Bsmt. remove the other feartures,
	#all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'] - all_data['BsmtUnfSF']
	all_data['Has_Bsmt'] = [1 if x>5 else 0 for x in all_data['TotalBsmtSF']]
	all_data['TotalBsmtFinSF'] = [(all_data['TotalBsmtSF'].tolist()[i] - all_data['BsmtUnfSF'].tolist()[i]) for i in range(len(all_data))]
	all_data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtUnfSF'], axis=1, inplace=True)

	#change pool area to boolean
	all_data['Has_Pool'] = [1 if x>5 else 0 for x in all_data['PoolArea']]
	all_data.drop(['PoolArea', 'PoolQC'], axis=1, inplace=True)

	# Transforming some numerical variables that are really categorical

	#PorchAreas
	all_data['Total_PorchArea'] = [(all_data['WoodDeckSF'].tolist()[i] + all_data['OpenPorchSF'].tolist()[i] +
                                all_data['EnclosedPorch'].tolist()[i] + all_data['3SsnPorch'].tolist()[i] +
                                all_data['ScreenPorch'].tolist()[i]) for i in range(len(all_data))]
	all_data['Total_PorchArea'] = [x**0.75/15 for x in all_data['Total_PorchArea']]
	all_data.drop(['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch'], axis=1, inplace=True)
	
	#MSSubClass=The building class
	all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

	#Changing OverallCond into a categorical variable
	#all_data['OverallCond'] = all_data['OverallCond'].astype(str)
	
	#Drop the MoSold and YrSold Column, since no significance was obtained from ANOVA test
	all_data.drop('MoSold', axis=1, inplace=True)
	all_data.drop('YrSold', axis=1, inplace=True)

	#substract YearRemodAdd by 1950 since all the value is larger or equal 1950. and drop YearBuilt,
	#since if no RemodAdd, it is same with the YearBuilt
	all_data['YearRemodAdd'] = [x-1950 for x in all_data['YearRemodAdd']]
	all_data.drop('YearBuilt', axis=1, inplace=True)

	return all_data

#this function is for labelling
def label_en(one_hot_df):
	#create dictionaries for encoding
	dict1 = dict(zip(['NA','Po','Fa','TA','Gd','Ex'], range(6)))
	dict2 = dict(zip(['None','No','Mn','Av','Gd'], range(5)))
	dict3 = dict(zip(['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'], range(7)))
	dict4 = dict(zip(['N','Y'], range(2)))
	dict5 = dict(zip(['Mix','FuseP', 'FuseF', 'FuseA','SBrkr'], range(5)))
	dict6 = dict(zip(['Sal','Sev', 'Maj2','Maj1','Mod','Min2','Min1','Typ'], range(8)))
	dict7 = dict(zip(['None','Po','Fa','TA','Gd','Ex'], range(6)))
	dict8 = dict(zip(['None','Unf','RFn','Fin'], range(4)))
	dict9 = dict(zip(['N','P','Y'], range(3)))
	dict10 = dict(zip(['None','MnWw','GdWo','MnPrv','GdPrv'], [0,1,2,1,2]))

	# label encode select columns
	#combine the above two features into one
	one_hot_df['ExterQual'] = one_hot_df['ExterQual'].map(lambda x:dict1[x]).astype(np.number)
	one_hot_df['ExterCond'] = one_hot_df['ExterCond'].map(lambda x:dict1[x]).astype(np.number)
	one_hot_df['ExterQuCo'] = [(one_hot_df['ExterQual'].tolist()[i] + one_hot_df['ExterCond'].tolist()[i]) for i in range(len(one_hot_df))]
	one_hot_df.drop(['ExterQual','ExterCond'], axis=1, inplace=True)

	one_hot_df['BsmtQual'] = one_hot_df['BsmtQual'].map(lambda x:dict7[x]).astype(np.number)
	one_hot_df['BsmtCond'] = one_hot_df['BsmtCond'].map(lambda x:dict7[x]).astype(np.number)
	#combine the above two features into one
	one_hot_df['BsmtQuCo'] = [(one_hot_df['BsmtQual'].tolist()[i] + one_hot_df['BsmtCond'].tolist()[i])/2 for i in range(len(one_hot_df))]
	one_hot_df.drop(['BsmtQual','BsmtCond'], axis=1, inplace=True)
	one_hot_df['BsmtExposure'] = one_hot_df['BsmtExposure'].map(lambda x:dict2[x]).astype(np.number)
	one_hot_df['BsmtFinType1'] = one_hot_df['BsmtFinType1'].map(lambda x:dict3[x]).astype(np.number)
	one_hot_df['BsmtFinType2'] = one_hot_df['BsmtFinType2'].map(lambda x:dict3[x]).astype(np.number)
	one_hot_df['HeatingQC'] = one_hot_df['HeatingQC'].map(lambda x:dict1[x]).astype(np.number)
	one_hot_df['KitchenQual'] = one_hot_df['KitchenQual'].map(lambda x:dict1[x]).astype(np.number)
	one_hot_df['Functional'] = one_hot_df['Functional'].map(lambda x:dict6[x]).astype(np.number)
	one_hot_df['FireplaceQu'] = one_hot_df['FireplaceQu'].map(lambda x:dict7[x]).astype(np.number)
	one_hot_df['GarageFinish'] = one_hot_df['GarageFinish'].map(lambda x:dict8[x]).astype(np.number)
	one_hot_df['GarageQual'] = one_hot_df['GarageQual'].map(lambda x:dict7[x]).astype(np.number)
	one_hot_df['PavedDrive'] = one_hot_df['PavedDrive'].map(lambda x:dict9[x]).astype(np.number)

	#make a new feature as flag for Garage
	one_hot_df['Has_Garage'] = [1 if x>0 else 0 for x in one_hot_df['GarageFinish']]
	
	#drop the GarageYrBlt, coz we already have yearbuilt
	one_hot_df.drop('GarageYrBlt', axis=1, inplace=True)

	return one_hot_df

def impute_multi_2(all_data):
	# PoolQC : data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.
	
	all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

	# MiscFeature : data description says NA means "no misc feature"
	all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

	# Alley : data description says NA means "no alley access"
	all_data["Alley"] = all_data["Alley"].fillna("None")

	# Fence : data description says NA means "no fence"
	all_data["Fence"] = all_data["Fence"].fillna("None")

	# FireplaceQu : data description says NA means "no fireplace"
	all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

	# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
	# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
	all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

	# GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
	for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
		all_data[col] = all_data[col].fillna('None')

	# GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
	for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'): # ??? Fine only if use as categorical feature
		all_data[col] = all_data[col].fillna(0)

	# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement
	for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
		all_data[col] = all_data[col].fillna(0)

	# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
	for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
		all_data[col] = all_data[col].fillna('None')

	# MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
	all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
	all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

	# MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'
	all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

	# Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
	all_data = all_data.drop(['Utilities'], axis=1)

	# Functional : data description says NA means typical
	all_data["Functional"] = all_data["Functional"].fillna("Typ")

	# Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
	all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

	# KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
	all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

	# Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
	all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
	all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

	# SaleType : Fill in again with most frequent which is "WD"
	all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

	# MSSubClass : Na most likely means No building class. We can replace missing values with None
	all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
	
	# Adding total sqfootage feature, add boolean feature for Bsmt. remove the other feartures,
	#all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'] - all_data['BsmtUnfSF']
	all_data['Has_Bsmt'] = [1 if x>5 else 0 for x in all_data['TotalBsmtSF']]
	all_data['TotalBsmtFinSF'] = [(all_data['TotalBsmtSF'].tolist()[i] - all_data['BsmtUnfSF'].tolist()[i]) for i in range(len(all_data))]
	all_data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtUnfSF'], axis=1, inplace=True)

	#change pool area to boolean
	all_data['Has_Pool'] = [1 if x>5 else 0 for x in all_data['PoolArea']]
	all_data.drop(['PoolArea', 'PoolQC'], axis=1, inplace=True)

	# Transforming some numerical variables that are really categorical

	#PorchAreas
	all_data['Total_PorchArea'] = [(all_data['WoodDeckSF'].tolist()[i] + all_data['OpenPorchSF'].tolist()[i] +
                                all_data['EnclosedPorch'].tolist()[i] + all_data['3SsnPorch'].tolist()[i] +
                                all_data['ScreenPorch'].tolist()[i]) for i in range(len(all_data))]
	all_data['Total_PorchArea'] = [x**0.75/15 for x in all_data['Total_PorchArea']]
	all_data.drop(['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch'], axis=1, inplace=True)
	
	#MSSubClass=The building class
	all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

	#Changing OverallCond into a categorical variable
	#all_data['OverallCond'] = all_data['OverallCond'].astype(str)
	
	#Drop the MoSold and YrSold Column, since no significance was obtained from ANOVA test
	all_data.drop('MoSold', axis=1, inplace=True)
	all_data.drop('YrSold', axis=1, inplace=True)

	#substract YearRemodAdd by 1950 since all the value is larger or equal 1950. and drop YearBuilt,
	#since if no RemodAdd, it is same with the YearBuilt
	all_data['YearRemodAdd'] = [x-1950 for x in all_data['YearRemodAdd']]
	all_data.drop('YearBuilt', axis=1, inplace=True)

	return all_data
