# Process large data sets in chunks to fit more data in memory
import pandas as pd

chunk_size = 10**6
with pd.read_csv(filename, chunksize=chunk_size) as reader:
    for chunk in reader:
        process(chunk)

#%%
# Set dtypes when importing data
# default will be int64 or float64
# not all data requires that much memory
import pandas as pd

data = pd.read_csv('~/repos/utils/diamonds.csv',
                   dtype={'carat': 'int32',
                          'cut': 'str',
                          'color': 'str',
                          'clarity': 'str',
                          'depth': 'int32',
                          'table': 'int32',
                          'price': 'int64',
                          'x': 'int32',
                          'y': 'int32',
                          'z': 'int32'})
#%%
# Alternate methods for basic operations
df.['column']  #is better than df.column b/c columns could have spaces or special characters

df.sum() # is much faster than sum(df) b/c it is optimized for data frames

isna # is better than isnull b/c it is not deprecated

#%%
# Reindex the data frame
df.reset_index()

# group by, aggregations, and pivots cause multi-indexes which are difficult to navigate

#%%
# Importing JSON data
max_level = 3
df = pd.read_json(filename) # Nested structure of data frame is preserved columns contain dictionaries
pd.json_normalize(df, max_level = max_level) # Automatically detects and flattens nested structures, control maximum depth

# Pandas groupby guide
# https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html

#%%
# Change display options when printing data frame
# Apply global options

max_rows = 1000  # too big & you will crash jupyter
max_columns = 20
max_colwidth = 20

#%%
# Plotting with pandas
df.plot()
# Options are 'line', 'bar'/'barh', 'hist', 'box', 'kde'/'density', 'area', 'scatter', 'hexbin', 'pie'
# Uses matplotlib on backed - can use matplotlib to further customize plots



