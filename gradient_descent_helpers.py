import numpy as np


def linear_fn(x_matrix, w_vect, b):
  '''
  The Fwb(x) function - Computes dependent variable y using matrix multiplication

  Parameters:
    x_matrix (ndarray (m,n))
    w_vect (ndarray (n,))
    b (scalar)
  
  Returns:
    y-prediction (vector)
  '''
  y_pred = x_matrix @ w_vect + b
  return y_pred


def logistic_fn(x_matrix, w_vect, b):
    z = linear_fn(x_matrix, w_vect, b)
    y_vect = 1 / (1 + np.exp(-z))
    return y_vect


def cost_fn_linear(x_matrix, y_vect, w_vect, b, lambda_reg = 0.0):
  '''
  Returns the cost.

  Parameters:
    x_matrix (ndarray (m,n))
    y_vect (ndarray (m,))
    w_vect (ndarray (n,))
    b (scalar)
    lambda_reg (scalar): regularization coeff
  
  Returns:
    cost (scalar)
  '''
  m = x_matrix.shape[0]
  y_pred = linear_fn(x_matrix, w_vect,b)
  error = (y_pred - y_vect)**2
  error_sum = error.sum()
  cost = error_sum / (2*m)

  #calc regularization value
  if(lambda_reg > 0):
    reg_term = (lambda_reg / (2*m)) * np.sum(w_vect**2)
    cost += reg_term
  return cost


def cost_fn_logistic(x_matrix, y_vect, w_vect, b, lambda_reg = 0.0):
    y_pred_vect = logistic_fn(x_matrix, w_vect, b)
    loss_vect = -y_vect * np.log(y_pred_vect) - (1-y_vect) * np.log(1-y_pred_vect)
    total_loss = np.sum(loss_vect)

    m = x_matrix.shape[0]
    cost = total_loss / m

    #calc regularization value
    if(lambda_reg > 0):
      reg_term = (lambda_reg / (2*m)) * np.sum(w_vect**2)
      cost += reg_term

    return cost


def grad_fn(x_matrix, y_vect, w_vect_input, b_input, predict_fn, lambda_reg = 0.0):
  '''
  Computes gradient(partial derivative) for w and b.

  Parameters:
    x_matrix (ndarray (m,n))
    y_vect (ndarray (m,))
    w_vect_input (ndarray (n,)): w-input
    b_input (scalar): b-input
    predict_fn (fn): function to predict dependent variables
    lambda_reg (scalar): regularization coeff
  
  Returns:
    w_vect (ndarray (n,)): w-output
    b (scalar): b-output
  '''
  m, n = x_matrix.shape
  b = 0.0
  w_vect = np.zeros(n)
  y_pred = predict_fn(x_matrix, w_vect_input, b_input)          
  error = y_pred - y_vect               
  w_vect = (1/m) * (x_matrix.T @ error)  

  if(lambda_reg > 0):
    reg_val_vect = (lambda_reg / m) * w_vect_input
    w_vect += reg_val_vect
  
  b = (1/m) * np.sum(error)    
  return w_vect, b


def grad_fn_linear(x_matrix, y_vect, w_vect_input, b_input, lambda_reg = 0.0):
  gradient = grad_fn(x_matrix, y_vect, w_vect_input, b_input, linear_fn, lambda_reg)
  return gradient

def grad_fn_logistic(x_matrix, y_vect, w_vect_input, b_input, lambda_reg = 0.0):
  gradient = grad_fn(x_matrix, y_vect, w_vect_input, b_input, logistic_fn, lambda_reg)
  return gradient


def gradient_descent(x_matrix, y_vect, w_vect_init, b_init, cost_fn, grad_fn, epochs, lr, lambda_reg = 0.0):
  '''
  Performs gradient descent.

  Parameters:
    x_matrix (ndarray (m,n))
    y_vect (ndarray (m,))
    w_vect_init (ndarray (n,)): initial w
    b_init (scalar): initial b
    cost_fn (function): cost function to be used
    grad_fn (function): gradient function to be used
    epochs (int): number of iterations
    lr (float): learning rate
  
  Returns:
    w_vect (ndarray (n,))
    b (scalar)
    cost_history (list): list of cost for each iteration
  '''
  w_vect = w_vect_init
  b = b_init
  n = x_matrix.shape[1]
  cost_history = []
  for epoch in range(epochs):
    #get cost
    cost = cost_fn(x_matrix, y_vect, w_vect, b, lambda_reg)
    #get gradient
    w_vect_grad, b_grad = grad_fn(x_matrix, y_vect, w_vect, b, lambda_reg)
    #update parameters
    b = b - lr * b_grad
    w_vect = w_vect - lr * w_vect_grad
    cost_history.append(cost)

  return w_vect, b, cost_history


def run_linear_gd(x_matrix, y_vect, w_vect_init, b_init, epochs, lr, lambda_reg = 0.0):
  return gradient_descent(x_matrix, y_vect, w_vect_init, b_init, cost_fn_linear, grad_fn_linear, epochs, lr, lambda_reg)


def run_logistic_gd(x_matrix, y_vect, w_vect_init, b_init, epochs, lr, lambda_reg = 0.0):
  return gradient_descent(x_matrix, y_vect, w_vect_init, b_init, cost_fn_logistic, grad_fn_logistic, epochs, lr, lambda_reg)


def zscore(x_matrix, column_indexes=None):
  '''
  Computes z-score normalized matrix.

  Parameters:
    x_matrix (ndarry (m,n)): matrix to be normalized
    column_indexes (array_like): list of column indexes to be normalized
  Returns:
    x_matrix_z (ndarray (m,n))
  '''
  x_matrix_z = None
  if(column_indexes is None):
    x_matrix_z = np.zeros(x_matrix.shape)
    mean = np.mean(x_matrix, axis=0)
    std_dev = np.std(x_matrix, axis=0)
    x_matrix_z = (x_matrix - mean) / std_dev
  else:
    x_matrix_z = np.copy(x_matrix)
    for i in column_indexes:
      x_vect = x_matrix[:,i]
      x_vect_z = (x_vect - np.mean(x_vect)) / np.std(x_vect)
      x_matrix_z[:,i] = x_vect_z
  return x_matrix_z

def get_mae(y_pred_vect, y_vect):
  mae = np.mean(np.abs(y_pred_vect - y_vect))
  y_mean = np.mean(y_vect)
  mae_perc = (mae / y_mean) * 100
  return (mae, mae_perc)

def print_list(list, number_of_prints=10):
  n = len(list)

  #print
  print_step = int(n/number_of_prints)
  for i in range(n):
    if(i % print_step == 0):
      val = list[i]
      print(i,f"{val:.2e}")