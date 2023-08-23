import matplotlib.pyplot as plt

def plot_list(list, only_top=0):
  y_values = list if only_top == 0 else list[:only_top]
  x_values = range(len(y_values))
  plt.figure(figsize=(3, 3))
  plt.plot(x_values, y_values)
  plt.show()

def plot_matrix(X):
  rows = X.shape[0]
  columns = X.shape[1]
  fig, ax = plt.subplots(1, columns, figsize=(15, 3))
  for j in range(columns):
    ax[j].scatter(range(rows), X[:, j])

  plt.tight_layout()
  plt.show()
