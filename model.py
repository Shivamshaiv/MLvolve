import sys
sys.path.insert(0, ".//Agents")

import time
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation,SimultaneousActivation,StagedActivation
import torch
import pyro
import torch.distributions as tod
import pyro.distributions as pyd
import matplotlib.pyplot as plt
from scipy.stats import poisson
import plotly.graph_objects as go
from scipy import ndimage
from Agents.labs import Labs
from Agents.funding import Funding
from Agents.researchers import Student,Junior
from Agents.landscape import Episthemic_Landscape

class WorldModel(Model):
    """A model with some number of agents."""
    def __init__(self, N_students,N_juniors,num_labs,elsize = 100,funding_nos = 12,num_topics = 5,m_j = 25_000 , m_u = 12_000,lamb = 0.1,remove_thres = 5):
        self.timestep = 0                             # Start of time in the model
        self.num_agents_s = N_students
        self.num_agents_j = N_juniors
        self.num_labs = num_labs
        self.funding_nos = funding_nos
        self.elsize = elsize
        model_stages = ["step_stage_1", "step_stage_2","step_stage_3","step_stage_4", "step_stage_5","step_stage_6","step_stage_7","step_stage_final"]
        self.topics = num_topics     # We start with having 5 topics
        self.m_j = 25_000
        self.m_u = 12_000
        self.lamb = lamb
        self.remove_thres = remove_thres
        self.schedule = StagedActivation(self,model_stages)
        lab_arr = []
        # Create agents
        for _ in range(self.topics):
            self.schedule.add(Episthemic_Landscape(0,self,self.elsize))
        for _ in range(self.funding_nos):
            self.schedule.add(Funding(0,self))
        for _ in range(self.num_agents_s):
            self.schedule.add(Student(0,self))
        for _ in range(self.num_agents_j):
            self.schedule.add(Junior(0,self))
        for j in range(self.num_labs):
            c = Labs(j,self)
            lab_arr.append(c)
        #self.sorted_labs = sorted(lab_arr, key=lambda x: x.lab_repute, reverse=True)
        #print("The lenght of",len(sorted_labs))
        #for lab in self.sorted_labs:
            self.schedule.add(c)
            #print("Lab",lab,"is added")
        
    def plot_stats(self):
        fig, axs = plt.subplots(5,2)
        fig.suptitle("The overall bids and exploration in the landscapes")
        landscapes = [agent for agent in self.schedule.agents if (agent.category=='Elandscape')]
        for i in range(5):
          axs[i,0].plot(landscapes[i].num_wining_bids)
          axs[i,0].set_title("# of wining bids in"+str(landscapes[i].unique_id))
          axs[i,1].plot(landscapes[i].explored_rate)
          axs[i,1].set_title("Cells explored in"+str(landscapes[i].unique_id))
        plt.show()

    def step(self,to_print = True):
        '''Advance the model by one step.'''
        self.timestep+= 1
        print("Timestep:",self.timestep)
        for _ in range(self.funding_nos):
            self.schedule.add(Funding(0,self))
        self.schedule.step()
        senior_agent =  [agent for agent in self.schedule.agents if (agent.category == 'S'  )]
        sorted_senior_agent = sorted(senior_agent, key=lambda x: x.bid_value, reverse=True)
        sorted_senior_agent_repute = sorted(senior_agent, key=lambda x: x.reputation_points, reverse=True)
        sorted_senior_agent_fund = sorted(senior_agent, key=lambda x: x.funding, reverse=True)

        junior_agent =  [agent for agent in self.schedule.agents if (agent.category == 'J')]
        sorted_senior_agent = sorted(senior_agent, key=lambda x: x.bid_value, reverse=True)

        student_agent =  [agent for agent in self.schedule.agents if (agent.category == 'U')]
        sorted_student_agent = sorted(student_agent, key=lambda x: x.reputation, reverse=True)

        lab_agent =  [agent for agent in self.schedule.agents if (agent.category == 'lab')]
        sorted_lab_agent = sorted(lab_agent, key=lambda x: x.lab_repute, reverse=True)

        landscape_agent =  [agent for agent in self.schedule.agents if (agent.category == 'Elandscape')]

        print("--------")
        print("Highest funded senior researcher is ",sorted_senior_agent[0].unique_id)
        print("Highest reputed senior researcher is",sorted_senior_agent_repute[0].unique_id)
        print("Number of juniors ",len(junior_agent))
        print("Number of students",len(student_agent))
        for i in range(self.topics):
          print("The average value of significance in landscape",i,"is",round(np.mean(sum(landscape_agent[i].matrix)),5))
        print("--------")
        
        if to_print:
          
          for agent in sorted_senior_agent:
              agent.printing_step()

          print("=============")

        
          for agent in sorted_junior_agent:
              agent.printing_step()


          print("=============")

          
          for agent in sorted_student_agent:
              agent.printing_step()

          print("=============")

    
          for agent in sorted_lab_agent:
              agent.printing_step()



empty_model = WorldModel(N_students = 100,N_juniors = 100,num_labs = 10)
for _ in range(10):
  empty_model.step(to_print = False)

  print("-------------")
empty_model.plot_stats()