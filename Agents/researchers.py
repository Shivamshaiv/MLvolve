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
%matplotlib inline



class Student(Agent):
    """An agent who is a university student."""
    def __init__(self,unique_id,model):
        super().__init__(unique_id,model)
        self.model = model
        self.unique_id = 'U_' + str(len(model.schedule.agents) + 1)
        self.category = 'U'   # University Student Category
        self.iq = self.iq_gen()
        self.ambitions = self.goal_generator()
        self.location = self.location_generator()
        self.publications = self.gen_publications()
        self.citations = self.gen_citations()
        self.reputation_points = 0
        #self.reputation = self.compute_reputation(model)

    def iq_gen(self):
      '''
      Defines the IQ of the agent, this IQ is sampled from two gaussians G0 and G1
      here G0 = N(90,10) and G1 = N(110,10) with a joint Bernolli distribution of 0.7
      '''
      will_research = pyro.sample("will_research",pyd.Bernoulli(0.7))
      mean_iq = list([90,110])[int(will_research.item())]
      the_iq = pyro.sample("the_iq",pyd.Normal(mean_iq,10))
      return the_iq.item()

    def goal_generator(self): 
      '''
      Defines the weights one puts to evalaute one's research career.
      Inspired from https://stackoverflow.com/questions/5563808/how-to-generate-three-random-numbers-whose-sum-is-1
      '''
      a = pyro.sample("first_pivot",pyd.Uniform(0,1))
      b = pyro.sample("second_pivot",pyd.Uniform(0,1))
      con1 = np.array([0,0,1])
      con2 = a.item()*np.array([1,0,-1])
      con3 = b.item()*np.array([0,1,-1])
      final_importance = np.add(con1,con2)
      final_importance = np.add(final_importance,con3)
      return final_importance

    def location_generator(self):
      '''
      Define's wether the person is in North America or Europe
      From : https://jfgagne.ai/talent/
      '''
      people_usa = 9010
      people_canada = 1154
      people_america = people_usa + people_canada
      people_europe = 1861 + 626 + 797 + 606
      euro_prob = 1/(1+(people_america/people_europe))
      loca = pyro.sample("location_",pyd.Bernoulli(euro_prob))
      location = 'eu' if loca.item() == 1.0 else 'am'
      return location

    def namegen(self,prefix):
      return prefix + "_" + str(self.unique_id)

    def gen_publications(self):
      num_pub = pyro.sample(self.namegen("number_publication"),pyd.Poisson(self.iq/100.))
      return num_pub.item()

    def gen_citations(self):
      h_index_computed = pyro.sample(self.namegen("h_index"),pyd.Normal(3,0.3))
      h_index = h_index_computed.item()
      if h_index < 0:
        h_index = 0
      return h_index*self.publications

    def gen_reputation_points(self,model):
      '''
      Reputation points are computed every tick and weighted sum of your citations, research experience and imperfectly your IQ
      '''
      w_citations = 1.5
      w_publications = 1.2
      w_iq = pyro.sample(self.namegen("how_iq_matters"),pyd.Normal(1,0.05)).item()

      agent_list = model.schedule.agents
      cit_list = [agent.citations for agent in agent_list if agent.category == 'U']
      pub_list = [agent.publications for agent in agent_list if agent.category == 'U']
      iq_list  = [agent.iq for agent in agent_list if agent.category == 'U']

      rep_points = (w_citations*(self.citations/max(cit_list))+ w_publications*(self.publications/max(pub_list)) + w_iq*(self.iq/max(iq_list)))/(w_citations+w_publications+w_iq)
      return rep_points


    def compute_reputation(self,model):
      agent_list = model.schedule.agents
      rep_list = []
      for agent in agent_list:
        if agent.category == 'U':
          rep_list.append(agent.reputation_points)
      sorted_rep = sorted(rep_list)
      return sorted_rep.index(self.reputation_points)


    def step_stage_1(self):
      self.reputation_points = self.gen_reputation_points(self.model)

    def step_stage_2(self):
      self.reputation = self.compute_reputation(self.model)

    def step_stage_final(self):
      print(self.unique_id,"has a reputation of ",self.reputation,"because thier points are",self.reputation_points)

    def step(self):
      pass
      #self.reputation_points = self.gen_reputation_points(self.model)
      #self.reputation = self.compute_reputation(self.model)

    def advance(self):
      print(self.unique_id,"has a reputation of ",self.reputation,"because thier points are",self.reputation_points)



class WorldModel(Model):
    """A model with some number of agents."""
    def __init__(self, N):
        self.num_agents = N
        model_stages = ["step_stage_1", "step_stage_2","step_stage_final"]
        self.schedule = StagedActivation(self,model_stages)
        # Create agents
        for i in range(self.num_agents):
            a = Student(i,self)
            self.schedule.add(a)

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()

