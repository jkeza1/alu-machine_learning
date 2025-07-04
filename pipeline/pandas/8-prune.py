#!/usr/bin/env python3
"""
This module contains a script that prunes a pd.DataFrame.
by removing the rows with missing values.
aka NaN values.
"""

import pandas as pd


from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.dropna()

print(df.head())