import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
import gradient_descent_helpers as gd
import plot_helpers as plth

x_matrix = np.arange(12).reshape(-1, 3).astype(float)
print(x_matrix)

print(gd.zscore(x_matrix, [0,2]))