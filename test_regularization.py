import numpy as np
import matplotlib.pyplot as plt
import plot_helpers as plth
import gradient_descent_helpers as gd
import load_data_helpers as data

x_vect = np.array(range(10))
rng = np.random.default_rng(1)
noise = rng.uniform(0, 15, x_vect.shape)
y_vect = x_vect ** 2 + noise

x = x_vect
x_vect_list=[]
for i in range(1, 15):
    x_vect_list.append(x**i)
x_matrix = np.column_stack(x_vect_list)
x_matrix, _, _ = gd.zscore(x_matrix)
w_init = np.zeros(x_matrix.shape[1])
b_init = 0.0

#without regularization
w_vector0, b0, cost_history0 = gd.run_linear_gd(x_matrix, y_vect, w_init, b_init, 100_000, 0.01, 0)

#with regularization
w_vector1, b1, cost_history1 = gd.run_linear_gd(x_matrix, y_vect, w_init, b_init, 100_000, 0.01, 1)

#print
with np.printoptions(suppress=True, precision=2, ):
    print(np.column_stack((w_vector0, w_vector1)))
    print(b0, b1)
    print(cost_history0[-1], cost_history1[-1])

plth.plot_matrix(np.column_stack((w_vector0, w_vector1)))

y_pred = np.dot(x_matrix,w_vector1)  + b1
plt.scatter(x_vect, y_vect)
plt.plot(x_vect, y_pred, color='red')
plt.show()