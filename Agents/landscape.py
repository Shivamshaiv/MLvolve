
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
import plotly.express as px
from utils.landscape_utils import bivariate_normal,get_peak,add_gaussian,remove_gussian_peak,top_k_2d_array



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
      self.explored = np.zeros([self.size,self.size])
      self.num_peaks = number_of_peaks
      self.simulate_guassian_peaks(self.num_peaks,width = 7 , height = 3)
      self.x_arr = []    # Explored corrdinates on the landscape
      self.y_arr = []    # Explored corrdinates on the landscape
      self.visible_x = []
      self.visible_y = []
      self.C_ = np.linspace(2_0000,7_0000,self.model.topics)[int(self.topic)]
      self.bid_win_count = 0
      self.bid_win_sig_store = []
      self.bid_win_store = []
      self.best_bid_store = []
      self.bid_store = np.zeros([self.size,self.size])
      self.all_bid_store = np.zeros([self.size,self.size])
      self.tot_sig = [np.sum(self.matrix)]
      self.num_wining_bids = []
      self.explored_rate = []
      self.colorscale=[[0.0, "rgb(255, 0, 0)"],[1.0, "rgb(255, 0, 0)"]]
      self.frame1  = go.Contour(z=self.matrix,colorscale='Greys',opacity = 0.65,visible = True,autocolorscale=False)
      self.frame2  = go.Heatmap(z=np.where(self.bid_store == 1,1,None),colorscale=self.colorscale,
        showscale = False,opacity = 1 )
      #self.frame2 = go.Scatter(x = [],y=[])
      self.frames = []

  
  def simulate_guassian_peaks(self,number,width = 7,height = 3):
    for _ in range(number):
      locx = np.random.randint(0,self.matrix.shape[0]-1)
      locy = np.random.randint(0,self.matrix.shape[1]-1)
      sig = width*np.random.normal(0.8,0.2)
      new_height = height*np.random.normal(1,0.2)
      #loc_track.append([locx,locy,sig,new_height])
      self.matrix = add_gaussian(self.matrix,self.size,[locx,locy],sig,new_height,self.max_height)
      self.diff_matrix = add_gaussian(self.diff_matrix,self.size,[locx,locy],sig,new_height,self.max_height)


  def add_gaussian_(self,matrix, loc, sig, height):  # height can be negative
      matrix += get_peak(self.size, loc, (sig, sig), height)
      # do not allow negative values
      matrix = matrix.clip(min=0, max=self.max_height)
      return matrix

  def reduce_novelty(self,loc,sucess_modifier):
      height = self.matrix[loc[0],loc[1]] * -(0.5*sucess_modifier)
      sig = self.size * 0.05
      self.matrix = self.add_gaussian_(self.matrix,loc,sig,height)

  def step_stage_1(self):
      self.original_mat = self.matrix

  def step_stage_2(self):
      pass

  def step_stage_3(self):
      pass

  def step_stage_4(self):
      pass

  def step_stage_5(self):
      pass

  def step_stage_6(self):
      pass

  def step_stage_7(self):
      pass

  def step_stage_final(self):
    self.tot_sig.append(np.sum(self.matrix))
    self.bid_win_store.append(sum(self.bid_win_sig_store))
    self.best_bid_store.append(sum(top_k_2d_array(self.original_mat,len(self.bid_win_sig_store))))
    temp_bider = np.where(self.bid_store == 1,1,None)
    self.frames.append(go.Frame(data = [go.Heatmap(z=temp_bider,colorscale=self.colorscale,showscale = False,opacity = 1),
      go.Contour(z=self.matrix,colorscale='Greys',opacity = 0.7,autocolorscale=False)],traces = [0,1]))
    #self.frames.append(go.Frame(data = [go.Contour(z=self.matrix,colorscale='Greys',opacity = 1,visible = True),
      #go.Scatter(x = np.where(self.bid_store == 1)[1],
      #y =np.where(self.bid_store == 1)[0],mode='markers',marker = dict(size = 5,line=dict(width=1,color='DarkSlateGrey')))]))

    tot_view = len(np.where(self.explored == 1)[1])
    num_wining_bids = len(np.where(self.bid_store == 1)[1])
    self.num_wining_bids.append(num_wining_bids)
    self.explored_rate.append(tot_view)
    self.bid_win_count = 0
    self.bid_win_sig_store = []
    to_plot = self.model.to_plot
    if self.model.timestep % int(self.model.plot_interval) == 0 and to_plot:

      plt.figure(figsize=(15,15))
      plt.suptitle(str(self.unique_id+'('+str(tot_view)+'/'+str(self.size**2)+')____'+str(num_wining_bids)))

      plt.subplot(121)
      plt.imshow(self.explored)
      plt.contour(self.matrix,alpha = 0.75)
      

      plt.subplot(122)
      plt.contour(self.matrix,alpha = 0.75)
      plt.imshow(self.bid_store)
      plt.show()
      

  def printing_step(self):
      #print(self.unique_id,"has employment of",self.employed,"and an affiliation of",self.affliation)
      print(self.unique_id,"has a reputation of ",self.reputation, ",cits are",self.citations,"and pubs are",self.publications,"and an affiliation of",self.affliation)
  

