#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:35:18 2022

@author: zeather
"""
import pandas as pd
import pickle
import os
import time

files = os.listdir('folder/')

loc = 'raw_data_location/'
out = 'output_location/'


for file in files:
    
    path = loc + file
    
    start_time = time.time()
    
    # any function or functions to extract features from a raw data set for ML
    
    features = build_features(path)
    
    out_path = out + file + '.pkl'
    pickle.dump(features, open(out_path, 'wb'))
    
    print('File:', file, 'feature extraction is complete after', (time.time()-start_time)/60,'minutes')
