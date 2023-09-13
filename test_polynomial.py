import numpy as np
import matplotlib.pyplot as plt
import plot_helpers as plth
import gradient_descent_helpers as gd
import load_data_helpers as data

# load the dataset
x_vector = np.arange(0,30)
y_vector = np.cos(x_vector/2)

#train
x = x_vector
x_matrix = np.column_stack((x, x**2, x**3, x**4, x**5, x**6))
x_matrix, _, _ = gd.zscore(x_matrix)
w_init = np.zeros(x_matrix.shape[1])
b_init = 0.0

w_vector, b, cost_history = gd.run_linear_gd(x_matrix, y_vector, w_init, b_init, 10_000, 0.2, 0.0)
print(w_vector, b, cost_history[-1])

y_pred = np.dot(x_matrix,w_vector)  + b
plt.scatter(x_vector, y_vector)
plt.plot(x_vector, y_pred, color='red')
plt.show()