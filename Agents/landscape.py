import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation,SimultaneousActivation,StagedActivation
import torch
import pyro
import torch.distributions as tod
import pyro.distributions as pyd
import matplotlib.pyplot as plt
from quantecon.distributions import BetaBinomial
from scipy.stats import poisson
import plotly.graph_objects as go




class Episthemic_Landscape(Agent):
  """The Episthemic Landscape of AI Feilds"""
  def __init__(self,unique_id,model,size = 100,number_of_peaks = 40):
      super().__init__(unique_id,model)
      self.model = model
      self.topic = len([agent for agent in model.schedule.agents if agent.category == 'Elandscape'])
      self.category = 'Elandscape'
      self.unique_id = 'Elandscape_' + str(self.topic)
      self.max_height = 10
      self.size = size
      self.matrix = np.random.rand(self.size, self.size)
      self.diff_matrix = np.random.rand(self.size, self.size)
      self.num_peaks = number_of_peaks
      self.simulate_guassian_peaks(self.num_peaks,width = 7 , height = 3)
      
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

  def add_gaussian(matrix, loc, sig, height):  # height can be negative
    matrix += self.get_peak(size, loc, (sig, sig), height)
    # do not allow negative values
    matrix = matrix.clip(min=0, max=max_height)
    return matrix

  def remove_gussian_peak(matrix,loc,size):
    height = matrix[loc[0],loc[1]]*-1    # Lower the current position by its height
    print("Height is",height)
    sig = size*0.01
    matrix = add_gaussian(matrix,loc,sig,height)
    return matrix

  def reduce_novelty(matrix,loc,size):
    height = matrix[loc[0],loc[1]] * -0.5
    print("Reduced novely height is",height)
    sig = size * 0.05
    matrix = add_gaussian(matrix,loc,sig,height)
    return matrix

  
  def simulate_guassian_peaks(self,number,width = 7,height = 3):
    for _ in range(number):
      locx = np.random.randint(0,matrix.shape[0]-1)
      locy = np.random.randint(0,matrix.shape[1]-1)
      sig = width*np.random.normal(0.8,0.2)
      new_height = height*np.random.normal(1,0.2)
      #loc_track.append([locx,locy,sig,new_height])
      self.matrix = add_gaussian(self.matrix,[locx,locy],sig,new_height)
      self.diff_matrix = add_gaussian(self.diff_matrix,[locx,locy],sig,new_height)

  def step_stage_1(self):
      #print("Epithemic Landscape",self.topic)
      pass

  def step_stage_2(self):
      pass

  def printing_step(self):
      #print(self.unique_id,"has employment of",self.employed,"and an affiliation of",self.affliation)
      print(self.unique_id,"has a reputation of ",self.reputation, ",cits are",self.citations,"and pubs are",self.publications,"and an affiliation of",self.affliation)
      #print("=======")

  def step_stage_3(self):
      pass

  def step_stage_final(self):
      pass
  

