import numpy as np

def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    This is a reimplementation of https://github.com/matplotlib/matplotlib/blob/81e8154dbba54ac1607b21b22984cabf7a6598fa/lib/matplotlib/mlab.py#L1866
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom

def get_peak(size, loc, sig, max_height = 5):
  
    '''Returns a 2d matrix of size*size with a peak centred at loc, with width sig, of max_height.'''

    m = np.mgrid[:size, :size]
    biv = bivariate_normal(m[0], m[1], sig[0], sig[1], loc[0], loc[1])
    return biv*float(max_height)/biv.max()

def add_gaussian(matrix,size, loc, sig, height,max_height):  # height can be negative
    matrix += get_peak(size, loc, (sig, sig), height)
    # do not allow negative values
    matrix = matrix.clip(min=0, max=max_height)
    return matrix

def remove_gussian_peak(matrix,loc,size,max_height):
  height = matrix[loc[0],loc[1]]*-1    # Lower the current position by its height
  print("Height is",height)
  sig = size*0.01
  matrix = add_gaussian(matrix,loc,sig,height,max_height)
  return matrix

def reduce_novelty(matrix,loc,size,max_height):
  height = matrix[loc[0],loc[1]] * -0.5
  print("Reduced novely height is",height)
  sig = size * 0.05
  matrix = add_gaussian(matrix,loc,sig,height,max_height)
  return matrix


def top_k_2d_array(arr,k):
  if np.ndim(arr) > 2:
    assert("Please pass an array with dimention less than 2")
  if np.ndim(arr) == 1:
    return arr[np.argsort(arr)[-k:]]
  else:
    flat_indices = np.argpartition(arr.ravel(), -k)[-k:]
    row_indices, col_indices = np.unravel_index(flat_indices, arr.shape)
    max_elements = arr[row_indices, col_indices]
    return max_elements