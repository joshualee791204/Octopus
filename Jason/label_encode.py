import pandas as pd
import numpy as np

def label_en(one_hot_df):
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
	one_hot_df['Has_Garage'] = [1 if x>0 else 0 for x in one_hot_df['TotalBsmtFinSF']]
	
	#drop the GarageYrBlt, coz we already have yearbuilt
	one_hot_df.drop('GarageYrBlt', axis=1, inplace=True)

	return one_hot_df