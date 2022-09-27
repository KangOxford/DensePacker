import pandas as pd
import numpy as np

csv = pd.read_csv("/mnt/Edisk/andrew/DensePacking-1/outcomes/analysis_test_cell_gym-v18.6.csv")
d = csv.drop(columns=csv.columns[[0]])
d1 = d.set_axis(['cell_penalty', 'packing_fraction'], axis=1)
d2 = d1[d1.cell_penalty <= 1e-2]
d3 = d2.sort_values(by = ['packing_fraction'])
d3.to_csv("/mnt/Edisk/andrew/DensePacking-1/outcomes/analysis-v18.6.csv")