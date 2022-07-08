# Split data set into features (independant) and outcome (dependant) variables

import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values        # select all rows and all columns except the last column   
y = dataset.iloc[:, -1].values         # select all rows and only the last column

# Impute missing values 

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, 1:3])                                                  # Include all numerical columns
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data

# Encoding the independant variable

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
print(X)

# Encoding the dependant variable

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Split features and outcomes into training and testing data sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    # 80/20 train/test split 
