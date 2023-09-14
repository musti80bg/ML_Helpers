import matplotlib.pyplot as plt
import math


def _get_ax(figsize=(3,3), equal_scale=False):
  fg = plt.figure(figsize=figsize)
  ax = fg.add_subplot()
  ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
  ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
  if(equal_scale):
    ax.axis('equal')
  return ax


def show_vect(y_vect, x_vect=None, figsize=(3,3), plot=True, scatter=False, equal_scale=False):
  ax = _get_ax(figsize=figsize, equal_scale=equal_scale)
  if(x_vect is None):
    x_vect = range(len(y_vect))

  if(plot):
    ax.plot(x_vect, y_vect)
  if(scatter):
    ax.scatter(x_vect, y_vect)

  plt.show()


def show_matrix(y_matrix, x_vect=None, figsize=(3,3), plot=True, scatter=False, equal_scale=False):
  ax = _get_ax(figsize=figsize, equal_scale=equal_scale)
  m, n = y_matrix.shape
  if(x_vect is None):
    x_vect = range(m)

  for j in range(n):
    y_vect = y_matrix[:, j]
    if(plot):
      ax.plot(x_vect, y_vect)
    if(scatter):
      ax.scatter(x_vect, y_vect)
  
  plt.show()


def show_matrix_sep(y_matrix, x_vect=None, figsize=(3,3), plot=True, scatter=False, equal_scale=False, ncols = 3):
  m, n = y_matrix.shape
  if(x_vect is None):
    x_vect = range(m)
  nrows = math.ceil(n / ncols)
  fig, axs = plt.subplots(nrows, ncols)
  axs = axs.reshape(-1)

  for j in range(n):
    ax = axs[j]
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    if(equal_scale):
      ax.axis('equal')

    y_vect = y_matrix[:, j]
    if(plot):
      ax.plot(x_vect, y_vect)
    if(scatter):
      ax.scatter(x_vect, y_vect)
  
  for d in range(n, len(axs)):
    ax_to_delete = axs[d]
    fig.delaxes(ax_to_delete)

  plt.tight_layout()
  plt.show()

