import pandas as pd
import numpy as np 


df = pd.read_csv('candles.csv')

print(df.head(5))

print(df.shape)

print(df.dtypes)