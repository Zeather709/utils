#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:14:31 2022

@author: zeather
"""

# This is fairly close to what R's summary() function does for pandas dataframes
# poor handling of categorical variable - to be updated

import pandas as pd

summary = df.describe(datetime_is_numeric = True, include = 'all')
print(summary)
