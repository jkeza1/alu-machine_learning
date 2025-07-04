#!/usr/bin/env python3
"""
This module contains a script that renames
the columns of a DataFrame.
"""

import pandas as pd


from_file = __import__('2-from_file').from_file

df = from_file('../Data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

#  Rename  Timestamp to Datetime
df = df.rename(columns={'Timestamp': 'Datetime'})

print(df.tail())