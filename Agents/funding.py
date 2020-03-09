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


class Funding(Agent):
  """The Funding Opportunies"""
  def __init__(self,unique_id,model):
      super().__init__(unique_id,model)
      self.model = model
      self.fund_no = len([agent for agent in model.schedule.agents if agent.category == 'fund'])
      self.category = 'fund'
      self.unique_id = 'fund_' + str(self.fund_no)
      self.grant_money = pyro.sample(self.namegen("funding_money"),pyd.Uniform(50000,20_00000)).item()
      self.capacity = 1     # The number of people to give the grant
      
  
  def namegen(self,prefix):
      return prefix + "_" + str(self.unique_id)
  


  def allocate_funding(self):
      other_funds =     [agent for agent in self.model.schedule.agents if (agent.category == 'fund')]
      sorted_funds = sorted(other_funds, key=lambda x: x.grant_money, reverse=True)
      own_position = sorted_funds.index(self)
      self.fund_rank = own_position
      
      sorted_fund_capacity = [agent.capacity for agent in sorted_funds]
      seniors_agents = [agent for agent in self.model.schedule.agents if (agent.category == 'S')]
      
      sorted_seniors = sorted(seniors_agents, key=lambda x: x.bid_value, reverse=True)
      #self.print_agent_list(sorted_juniors)
      #self.print_agent_list(sorted_labs)
      sum_till = int(sum(sorted_fund_capacity[:own_position]))
      #print(own_position,sum_till)
      #print(sum_till)
      if len(seniors_agents) < sum_till:
          return
      cap = int(self.capacity)
      for agent in sorted_seniors[sum_till:]:
          if cap > 0:
            agent.funded_once = True
            agent.is_funded = True
            agent.funding = int(self.grant_money)
            agent.funding_source = self.unique_id
            agent.landscape.bid_store[agent.pos_y,agent.pos_x] = 1
            agent.landscape.bid_win_count += 1
            agent.landscape.bid_win_sig_store.append(agent.landscape.matrix[agent.pos_y,agent.pos_x])
            #print(agent.unique_id,"is funded by",self.unique_id,"the is funded is",agent.is_funded)
            cap -= 1





  def step_stage_1(self):
      #print("Epithemic Landscape",self.topic)
      pass

  def step_stage_2(self):
      #self.allocate_funding()
      pass

  def printing_step(self):
      #print(self.unique_id,"has employment of",self.employed,"and an affiliation of",self.affliation)
      #print(self.unique_id,"has a reputation of ",self.reputation, ",cits are",self.citations,"and pubs are",self.publications,"and an affiliation of",self.affliation)
      #print("=======")
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
      self.model.schedule.remove(self)
  


class Recruiter(Agent):
  """The Recruiter of Funding Opportunies"""
  def __init__(self,unique_id,model):
      super().__init__(unique_id,model)
      self.model = model
      self.rec_no = len([agent for agent in model.schedule.agents if agent.category == 'rec'])
      self.category = 'rec'
      self.unique_id = 'rec_' + str(self.rec_no)
      
  
  def namegen(self,prefix):
      return prefix + "_" + str(self.unique_id)
  
  def pref_generator(self): 
      '''
      Defines the weights one puts to evalaute one's research career.
      Inspired from https://stackoverflow.com/questions/5563808/how-to-generate-three-random-numbers-whose-sum-is-1
      '''
      a = pyro.sample("first_pivot",pyd.Uniform(0,1))
      b = pyro.sample("second_pivot",pyd.Uniform(0,1-a.item()))
      con1 = np.array([0,0,1])
      con2 = a.item()*np.array([1,0,-1])
      con3 = b.item()*np.array([0,1,-1])
      final_importance = np.add(con1,con2)
      final_importance = np.add(final_importance,con3)
      return final_importance


  def allocate_funding(self):
      other_funds =     [agent for agent in self.model.schedule.agents if (agent.category == 'fund')]
      sorted_funds = sorted(other_funds, key=lambda x: x.grant_money, reverse=True)


      for fund in sorted_funds:
        seniors_agents = [agent for agent in self.model.schedule.agents if (agent.category == 'S' and not agent.is_funded)]
        pref = self.pref_generator()
        sorted_seniors = sorted(seniors_agents, key=lambda x: (pref[0]*x.current_bid + pref[1]*x.bid_novelty + pref[2]*x.reputation_points)/sum(pref), reverse=True)
        if len(sorted_seniors) < 1:
          break
        funded_this_time = sorted_seniors[0]     # Select the one which tops the list
        agent = funded_this_time
        agent.funded_once = True
        agent.is_funded = True
        agent.funding = int(fund.grant_money)
        agent.funding_source = fund.unique_id
        agent.times_funded += 1
        agent.landscape.bid_store[agent.pos_y,agent.pos_x] = 1
        agent.landscape.bid_win_count += 1
        agent.landscape.bid_win_sig_store.append(agent.landscape.matrix[agent.pos_y,agent.pos_x])


  def step_stage_1(self):
      #print("Epithemic Landscape",self.topic)
      pass

  def step_stage_2(self):
      self.allocate_funding()
      

  def printing_step(self):
      #print(self.unique_id,"has employment of",self.employed,"and an affiliation of",self.affliation)
      #print(self.unique_id,"has a reputation of ",self.reputation, ",cits are",self.citations,"and pubs are",self.publications,"and an affiliation of",self.affliation)
      #print("=======")
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
      pass