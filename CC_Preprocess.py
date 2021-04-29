#We'll use the Credit Card Approval dataset from the UCI Machine Learning Repository.(Prediction of Credit Approval)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Read the dataset
cc_apps=pd.read_csv('F:/MY FILE/Machine_Learning/Credit card Approval Assignment/cc_approvals.csv')
cc_apps.head()
# Print summary statistics
cc_apps.describe()
# Print DataFrame information
 #cc_apps.info()
# Inspect missing values in the dataset
print(cc_apps.isnull().values.sum())

# Replace the '?'s with NaN
cc_apps = cc_apps.replace("?",np.NaN)
# Inspect the missing values again
cc_apps.tail(10)

# Count the number of NaNs in the dataset to verify
print(cc_apps.isnull().sum())
##cc_apps = cc_apps.fillna(cc_apps.mode().iloc[0])

# Imputing missing observations in categorical columns with mode (alphabetically occurs)
for col in cc_apps:
 if  cc_apps[col].isnull().any():
     impute_values = cc_apps[col].value_counts().index[0]
     cc_apps[col].fillna(impute_values, inplace = True)

# Count the number of NaNs in the dataset and print the counts to verify
print(cc_apps.count())
cc_apps.tail(20)


"""cc_apps['A1'].value_counts()
cc_apps['A2'].value_counts()
cc_apps.count()    #to check the count in each column
#cc_apps_cat = cc_apps.copy()
#cc_apps_cat['A2'] = pd.qcut( cc_apps_cat['A2'], q=3, labels=['low', 'medium', 'high'])

cc_apps['A2'] .value_counts()

#Integer Encoding
level_mapping = {'low': 0, 'medium': 1, 'high': 2}

cc_apps_cat = cc_apps.copy()
cc_apps_cat['A2'] = cc_apps_cat['A2'].replace(level_mapping)
cc_apps_cat.head(5) """

####Credit card approvel data cleaning

# Preprocessing the data # Successfully converted all the non-numeric values to numeric ones.
le=LabelEncoder()
# Iterate over all the values of each column and extract their dtypes
for col in cc_apps:
    # Compare if the dtype is object
    if cc_apps[col].dtype=='object':
    # Use LabelEncoder to do the numeric transformation
         cc_apps[col]=le.fit_transform(cc_apps[col])
cc_apps.head()

# Now, let's try to understand what these scaled values mean in the real world.(-1 to +1)
# Segregate features and labels into separate variables
X = cc_apps.iloc[:,0:15].values
target = cc_apps.iloc[:,15].values

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# merge the dataset
df1 = pd.DataFrame(X)
df2 = pd.DataFrame(target)
df_clean = pd.concat([df1, df2], axis=1, ignore_index=True)

#### output clean the data
df_clean.to_csv("F:/MY FILE/Machine_Learning/Credit card Approval Assignment/df_clean.csv")

df_clean.shape
df_clean.describe(include='all').round(3) 
df_clean.head(5)


#################






