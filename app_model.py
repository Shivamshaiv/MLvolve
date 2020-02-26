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
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import ndimage
from Agents.labs import Labs
from Agents.funding import Funding
from Agents.researchers import Student,Junior
from Agents.landscape import Episthemic_Landscape
import streamlit as st

class WorldModel(Model):
    """A model with some number of agents."""
    def __init__(self, N_students,N_juniors,num_labs,elsize = 100,funding_nos = 12,num_topics = 5,
      m_j = 25_000 , m_u = 12_000,lamb = 0.1,remove_thres = 5,to_plot = True,plot_interval = 10,episilon = 0.05):
        self.timestep = 0                             # Start of time in the model
        self.time_arr = []
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
        self.to_plot = to_plot
        self.plot_interval = plot_interval
        self.episilon = episilon
        # Create agents
        for _ in range(self.topics):
            self.schedule.add(Episthemic_Landscape(0,self,self.elsize))
        for _ in range(self.funding_nos):
            self.schedule.add(Funding(0,self))
        for _ in range(self.num_agents_s):
            self.schedule.add(Student(0,self))
        for _ in range(self.num_agents_j):
            self.schedule.add(Junior(0,self))
        for _ in range(self.num_labs):
            self.schedule.add(Labs(0,self))

        
    def plot_stats(self):
        landscapes = [agent for agent in self.schedule.agents if (agent.category=='Elandscape')]
        subplot_name_arr = []
        for i in range(len(landscapes)):
          subplot_name_arr.append("# of wining bids: "+str(landscapes[i].unique_id[-1]))
          subplot_name_arr.append("Cells explored: "+str(landscapes[i].unique_id[-1]))
          subplot_name_arr.append("Significance gained: "+str(landscapes[i].unique_id[-1]))
          subplot_name_arr.append("Policy optimum "+str(landscapes[i].unique_id[-1]))
        bid_explore_plotly = make_subplots(rows = len(landscapes),cols = 4,shared_xaxes=True,subplot_titles = subplot_name_arr)
        for i in range(len(landscapes)):
          bid_explore_plotly.add_trace(
            go.Scatter(x=self.time_arr,y= landscapes[i].num_wining_bids,name = str(landscapes[i].unique_id),
              ), row = i+1,col = 1)

          bid_explore_plotly.add_trace(
            go.Scatter(x=self.time_arr,y= landscapes[i].explored_rate,name = str(landscapes[i].unique_id),
              ), row = i+1,col = 2)

          bid_explore_plotly.add_trace(
            go.Scatter(x=self.time_arr,y= np.abs(np.diff(landscapes[i].tot_sig)),name = str(landscapes[i].unique_id),
              ), row = i+1,col = 3)

          bid_explore_plotly.add_trace(
            go.Scatter(x=self.time_arr,y= landscapes[i].bid_win_store,name = str(landscapes[i].unique_id),
              ), row = i+1,col = 4)

          bid_explore_plotly.add_trace(
            go.Scatter(x=self.time_arr,y= landscapes[i].best_bid_store,name = "optimum"+str(landscapes[i].unique_id[-1]),opacity = 0.5,
              ), row = i+1,col = 4)


          #bid_explore_plotly.update_yaxes(
            #range(0,int(np.min(np.diff(landscapes[i].tot_sig)))-10),row = i+i,col =3 
            #)

        bid_explore_plotly.update_layout(showlegend=False,height=900, width=800,title_text="<b>The number of wining bids and exploration of landscapes</b>")
        st.plotly_chart(bid_explore_plotly)
        for i in range(len(landscapes)):
          plotly_f = go.Figure(data = [landscapes[i].frame2,landscapes[i].frame1],layout=go.Layout(title="Episthemic Landscape of topic "+str(landscapes[i].topic),
            updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None,{"frame": {"redraw": True},"fromcurrent": True, "transition": {"duration": 50}}])])]), frames = landscapes[i].frames)
          st.plotly_chart(plotly_f)

        if self.to_plot:
          fig, axs = plt.subplots(5,2,figsize=(20,20))
          fig.suptitle("The overall bids and exploration in the landscapes")
          for i in range(len(landscapes)):
            axs[i,0].plot(landscapes[i].num_wining_bids)
            axs[i,0].set_title("# of wining bids in"+str(landscapes[i].unique_id))
            axs[i,1].plot(landscapes[i].explored_rate)
            axs[i,1].set_title("Cells explored in"+str(landscapes[i].unique_id))
          plt.show()


    def step(self,to_print = True):
        '''Advance the model by one step.'''
        self.timestep+= 1
        self.time_arr.append(self.timestep)
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



st.title("Mlvolve : Agent based exploration of AI Research")
st.sidebar.markdown("## Initial Values")
no_timesteps_st = st.sidebar.number_input("# of timesteps",1,100,20)
no_student_st = st.sidebar.number_input("Number of students",1,1000,100)
no_juniors_st = st.sidebar.number_input("Number of juniors",1,1000,100)
no_labs_st    = st.sidebar.number_input("Number of labs",1,500,40)
no_funding_st = st.sidebar.number_input("# of funding opening",1,500,30)
st.sidebar.markdown("## Fine Tuning")
episilon_st =   st.sidebar.slider("Select an epsilon value",0.0,1.0,0.05,step = 0.05)

start_button_st = st.sidebar.button("Simulate")

if start_button_st:
  my_bar = st.progress(0)
  empty_model = WorldModel(
    N_students = no_student_st,
    N_juniors = no_juniors_st,
    num_labs = no_labs_st,
    funding_nos = no_funding_st,
    episilon = episilon_st,
    to_plot = False)
  my_placeholder = st.empty()
  for timer in range(no_timesteps_st):
    my_placeholder.text('Simulating timestep: '+str(timer+1))
    my_bar.progress(((timer+1)/no_timesteps_st))
    empty_model.step(to_print = False)

    print("-------------")
  empty_model.plot_stats()