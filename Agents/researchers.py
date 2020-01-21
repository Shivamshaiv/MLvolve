import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
import torch
import pyro
import torch.distributions as tod
import pyro.distributions as pyd
import matplotlib.pyplot as plt


class Student(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self,unique_id,model):
        super().__init__(unique_id,model)
        self.unique_id = len(model.schedule.agents) + 1
        self.category = 'S'   # Student Category
        self.iq = self.iq_gen()
        self.ambitions = self.goal_generator()
        self.location = self.location_generator()
        self.publications = 0
        self.citations = 0
        self.reputation = -1

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

    def step(self):
      print(self.unique_id)
    


class WorldModel(Model):
    """A model with some number of agents."""
    def __init__(self, N):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            a = Student(i,self)
            self.schedule.add(a)

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()

