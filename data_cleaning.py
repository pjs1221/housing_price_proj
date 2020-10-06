# housing_price_proj

import pandas as pd
import numpy as np
import math

df = pd.read_csv('Data/housing_ca.csv')

df = df.drop(columns=['id','url','region_url','image_url'])

df = df.drop_duplicates(subset=['price','long','lat'])

# Fix NaN values

df['laundry_options'] = df['laundry_options'].fillna('na')
df['parking_options'] = df['parking_options'].fillna('na')

df = df.dropna(subset=['lat', 'long'])


#Output cleaned data to new csv
df_out = df

#Output cleaned data to new csv
df_out.to_csv('Data/housing_ca_cleaned.csv',index = False)
