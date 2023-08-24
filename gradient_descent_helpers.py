import numpy as np


def zscore(x_matrix):
  '''
  Computes z-score normalized matrix.

  Parameters:
    x_matrix (ndarry (m,n)): matrix to be normalized

  Returns:
    x_matrix_z (ndarray (m,n))
    mean (ndarray (n,))
    std_dev (ndarray (n,))
  '''
  x_matrix_z = np.zeros(x_matrix.shape)
  mean = np.mean(x_matrix, axis=0)
  std_dev = np.std(x_matrix, axis=0)
  
  x_matrix_z = (x_matrix - mean) / std_dev
  return (x_matrix_z, mean, std_dev)


def f_fn(x_vector, w_vector, b):
  '''
  The Fwb(x) function - Computes dependent variable y

  Parameters:
    x_vector (ndarray (n,))
    w_vector (ndarray (n,))
    b (scalar)
  
  Returns:
    y-prediction (scalar)
  '''
  y_pred = np.dot(x_vector, w_vector) + b
  return y_pred


def cost_fn(x_matrix, y_vector, w_vector, b):
  '''
  Returns the cost.

  Parameters:
    x_matrix (ndarray (m,n))
    y_vector (ndarray (m,))
    w_vector (ndarray (n,))
    b (scalar)
  
  Returns:
    cost (scalar)
  '''
  m = x_matrix.shape[0]
  w_matrix = w_vector.reshape(-1, 1)
  y_pred = x_matrix @ w_matrix + b
  y_pred = y_pred[:,0]
  error = (y_pred - y_vector)**2
  error_sum = error.sum()
  cost = error_sum / (2*m)
  return cost


def grad_fn(x_matrix, y_vector, w_vector_input, b_input):
  '''
  Computes gradient(partial derivative) for w and b.

  Parameters:
    x_matrix (ndarray (m,n))
    y_vector (ndarray (m,))
    w_vector_input (ndarray (n,)): w-input
    b_input (scalar): b-input
  
  Returns:
    w_vector (ndarray (n,)): w-output
    b (scalar): b-output
  '''
  m, n = x_matrix.shape
  b = 0.0
  w_vector = np.zeros(n)
  for i in range(m):
    x_matrix_i = x_matrix[i]
    y_i = y_vector[i]
    y_pred_i = f_fn(x_matrix_i, w_vector_input, b_input)
    error_i = y_pred_i - y_i
    b = b + error_i
    for j in range(n):
      w_vector[j] = w_vector[j] + error_i * x_matrix_i[j]

  b = b / m
  w_vector = w_vector / m
  return w_vector, b


def gradient_descent(x_matrix, y_vector, w_vector_init, b_init, epochs, lr):
  '''
  Performs gradient descent.

  Parameters:
    x_matrix (ndarray (m,n))
    y_vector (ndarray (m,))
    w_vector_init (ndarray (n,)): initial w
    b_init (scalar): initial b
    epochs (int): number of iterations
    lr (float): learning rate
  
  Returns:
    w_vector (ndarray (n,))
    b (scalar)
    cost_history (list): list of cost for each iteration
  '''
  w_vector = w_vector_init
  b = b_init
  m = x_matrix.shape[0]
  n = x_matrix.shape[1]
  cost_history = []
  for epoch in range(epochs):
    cost = cost_fn(x_matrix, y_vector, w_vector, b)
    if(np.isnan(cost)):
      print(f"cost is nan for epoch {epoch}")
      break
    w_vector_grad, b_grad = grad_fn(x_matrix, y_vector, w_vector, b)
    b = b - lr * b_grad
    for j in range(n):
      w_vector[j] = w_vector[j] - lr * w_vector_grad[j]
    cost_history.append(cost)

  return w_vector, b, cost_history


def print_list(list, number_of_prints=10):
  n = len(list)

  #print
  print_step = int(n/number_of_prints)
  for i in range(n):
    if(i % print_step == 0):
      val = list[i]
      print(i,f"{val:.2e}")