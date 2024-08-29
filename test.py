import os

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt
from itertools import combinations

# read indexed_points_dia.csv and indexed_points_sys.csv with headers
df_dia = pd.read_csv('indexed_points_dia.csv', header=0)
df_sys = pd.read_csv('indexed_points_sys.csv', header=0)

print(df_dia.head(50))
print(df_sys.head(50))