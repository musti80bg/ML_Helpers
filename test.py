import numpy as np
import gradient_descent_helpers as gd
import load_data_helpers as data
import plot_helpers as plth

# # load the dataset
# x_train, y_train = data.load_house_data()
# x_features = ['size(sqft)','bedrooms','floors','age']
# b_init = 0.5
# w_init = np.array([0.5, 0.5, 0.5, 0.5])
# plth.plot_matrix(x_train)

# #train
# w_vector, b, cost_history = gd.gradient_descent(x_train, y_train, w_init, b_init, 1000, 0.00000091)
# plth.plot_list(cost_history)

x_vector = np.arange(0,5)
y_vector = x_vector**2
x_matrix = np.column_stack((x_vector, x_vector**2))
w_vector = np.ones(x_matrix.shape[1])
b = 0.0

m = x_matrix.shape[0]
w_matrix = w_vector.reshape(-1, 1)
y_pred = x_matrix @ w_matrix + b
y_pred = y_pred[:,0]
error = (y_pred - y_vector)**2
error_sum = error.sum()
cost = error_sum / (2*m)
print(cost)