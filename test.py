import numpy as np
import gradient_descent_helpers as gd
import load_data_helpers as data
import plot_helpers as plth

# load the dataset
x_train, y_train = data.load_house_data()
x_features = ['size(sqft)','bedrooms','floors','age']
b_init = 0.5
w_init = np.array([0.5, 0.5, 0.5, 0.5])
plth.plot_matrix(x_train)

#train
w_vector, b, cost_history = gd.gradient_descent(x_train, y_train, w_init, b_init, 1000, 0.00000091)
plth.plot_list(cost_history)