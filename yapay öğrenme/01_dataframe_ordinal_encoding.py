# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:14:14 2021

@author: cagda
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# verinizin ham hali
player_id=np.random.randint(1,10000,100)
df = pd.DataFrame(player_id, columns = ['Player'])
print(df.dtypes)

# verinin kategorik kodlanması için ön ek ekleme 
df['Player'] = 'p_' + df['Player'].astype(str)
print(df.dtypes)

# verinin kodlanması
encoder = OrdinalEncoder()
result = encoder.fit_transform(df)
df['Player']=result
df['Player']= df['Player'].astype('category')
print(df.dtypes)