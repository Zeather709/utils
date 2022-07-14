#usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:14:31 2022

@author: zeather
"""

# This is fairly close to what R's summary() function does for pandas dataframes
# poor handling of categorical variable - to be updated

import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('/home/zeather/repos/utils/diamonds.csv')
# Drop first column (optional)
df = df.iloc[:,1:]
df

# This is ok and quicker

summary = df.describe(datetime_is_numeric = True, include = 'all')
summary

# Much more robust solution, perhaps a bit excessive for some applications
# It takes quite a long time on larger data sets

profile = ProfileReport(df, title = 'Data Profile')

# Accessing values from the correlation matrices

correlations = profile.description_set["correlations"]
print(correlations.keys())
pearson_df = correlations["pearson"]
pearson_mat = pearson_df.values
print(pearson_mat)

# Export as html file 
profile.to_file("/home/zeather/Downloads/your_report.html")

# Display in Jupyter Notebook as widget
profile.to_widgets()

# Embed directly in notebook cell 
profile.to_notebook_iframe()