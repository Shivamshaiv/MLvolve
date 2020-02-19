import sys
sys.path.insert(0, ".")
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
from researchers import Senior


class Labs(Agent):
  """An agent who is a junior researcher PhD, postdoc , early Researchers"""
  def __init__(self,unique_id,model):
      super().__init__(unique_id,model)
      self.model = model
      self.unique_id = 'Lab_' + str(len([agent for agent in model.schedule.agents if agent.category == 'lab']) + 1)
      self.category = self.give_category()
      self.numtopics = 5
      self.capacity_s_initial = 0
      self.capacity_s = 0
      self.init_seniors(3)
      

  def init_seniors(self,num_seniors):
      for _ in range(num_seniors):
        a = Senior(0,self.model,self.unique_id)
        self.model.schedule.add(a)


  def namegen(self,prefix):
      return prefix + "_" + str(self.unique_id)

  def give_category(self):
      return "lab"

  def compute_capacity_j(self):
      capacity_j = pyro.sample(self.namegen("capcity_j"),pyd.Poisson(2)).item() + 1
      return capacity_j

  def compute_capacity_s(self):
      '''Compute the number of studetns to be recruited'''
      capacity_s = pyro.sample(self.namegen("capcity_s"),pyd.Poisson(3)).item() + 1
      return capacity_s

  def gen_lab_repute(self):
      senior_reputation = [agent.reputation_points for agent in self.model.schedule.agents if (agent.category == 'S' and agent.affliation == self.unique_id)]
      sorted_repu = sorted(senior_reputation)
      return (sorted_repu[-1] + sorted_repu[-2]+sum(sorted_repu[:-2]))/(2+(0.5*len(sorted_repu[:-2])))

  def print_agent_list(self,agent_list):
      list_name = [agent.unique_id for agent in agent_list]
      print(list_name)
  '''
  def recruit_juniors(self):
      other_labs =     [agent for agent in self.model.schedule.agents if (agent.category == 'lab')]
      sorted_labs = sorted(other_labs, key=lambda x: x.lab_repute, reverse=True)
      sorted_lab_capacity = [agent.capacity_j_initial for agent in sorted_labs]
      own_position = sorted_labs.index(self)
      self.sorted_labs = sorted_labs  # Stored to in the variabe to remove reduntant computations
      self.lab_rank = own_position
      juniors_agents = [agent for agent in self.model.schedule.agents if (agent.category == 'J')]
      
      sorted_juniors = sorted(juniors_agents, key=lambda x: x.reputation_points, reverse=True)
      #self.print_agent_list(sorted_juniors)
      #self.print_agent_list(sorted_labs)
      sum_till = int(sum(sorted_lab_capacity[:own_position]))
      cap = int(self.capacity_j)
      #print(sum_till)
      if len(juniors_agents) < sum_till:
          return
      for agent in sorted_juniors[sum_till:]:
          if self.capacity_j > 0:
            agent.employed = True
            agent.affliation = self.unique_id
            self.capacity_j -= 1
      '''


  def recruit_students(self):
      '''Function with calls the recuitment method of students'''
      other_labs =     [agent for agent in self.model.schedule.agents if (agent.category == 'lab')]
      sorted_labs = sorted(other_labs, key=lambda x: x.lab_repute, reverse=True)
      own_position = sorted_labs.index(self)
      self.lab_rank = own_position
      sorted_lab_capacity = [agent.capacity_s_initial for agent in sorted_labs]
      # self.lab_rank = own_position    #Already done
      student_agents = [agent for agent in self.model.schedule.agents if (agent.category == 'U')]
      
      sorted_student = sorted(student_agents, key=lambda x: x.reputation_points, reverse=True)
      #self.print_agent_list(sorted_juniors)
      #self.print_agent_list(sorted_labs)
      sum_till = int(sum(sorted_lab_capacity[:own_position]))
      #print(sum_till)
      cap = int(self.capacity_s)
      if len(student_agents) < sum_till:
          return
      for agent in sorted_student[sum_till:]:
          if self.capacity_s > 0:
            agent.employed = True
            agent.affliation = self.unique_id
            self.capacity_s -= 1
    
  def step_stage_1(self):
      #print("Reputation of lab is",self.lab_repute)
      #pass
      self.lab_repute = self.gen_lab_repute()

  def step_stage_2(self):
      #self.lab_repute = self.gen_lab_repute()
      seniors_recurited_list = [agent for agent in self.model.schedule.agents if (agent.category == 'S' and agent.affliation == self.unique_id)]
      for senior in seniors_recurited_list:
        senior.lab_coupled_reputation = (senior.reputation_points*(0.5)) + (self.lab_repute*(0.5))


  def step_stage_3(self):
      #self.recruit_juniors()
      pass

  def step_stage_4(self):
      self.recruit_students()

  def step_stage_5(self):
      pass

  def step_stage_6(self):
      pass

  def step_stage_7(self):
      self.capacity_s_initial = 0
      self.capacity_s = 0
      pass

  def printing_step(self):
      print("Reputation Rank of lab is",self.lab_rank,"Vacancy in",self.unique_id," is of students:",self.capacity_s,self.capacity_s_initial)

  def step_stage_final(self):
      pass