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

    def promote(self):
      promotion_dict = {'iq':self.iq , 'ambitions':self.ambitions,'location':self.location,'publications':self.publications,'citations':self.citations,"reputation_points":self.reputation_points}
      return promotion_dict

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







## Todo : Add Salary and money component
class Junior(Agent):
  """An agent who is a junior researcher PhD, postdoc , early Researchers"""
  def __init__(self,unique_id,model,promoted_attrs = None):
      super().__init__(unique_id,model)
      self.unique_id = 'J_' + str(len(model.schedule.agents) + 1)
      self.category = 'J'   # Junior Researcher Category
      self.numtopics = 5
      #self.topic_interested = self.select_topic()                   # We assume that there are 5 topics
      if promoted_attrs:
        self.iq = promoted_attrs['iq']
        self.location = promoted_attrs['location']
        self.publications =  promoted_attrs['publications']
        self.citations = promoted_attrs['citations']
        self.ambitions = promoted_attrs['ambitions']
        self.reputation_points = promoted_attrs['reputation_points']
        #self.reputation = self.compute_reputation()
      else:
        self.iq = self.iq_gen()
        self.location = self.location_generator()
        self.publications = self.gen_publications()
        self.citations = self.gen_citations()
        self.ambitions = self.goal_generator()
        self.reputation_points = 0
        

  def namegen(self,prefix):
      return prefix + "_" + str(self.unique_id)
  
  def iq_gen(self):
      '''
      Defines the IQ of the agent, this IQ is sampled from two gaussians G0 and G1
      here G0 = N(95,10) and G1 = N(115,10) with a joint Bernolli distribution of 0.95 for Junior Researchers
      '''
      will_research = pyro.sample("will_research",pyd.Bernoulli(0.95))
      mean_iq = list([95,115])[int(will_research.item())]
      the_iq = pyro.sample("the_iq",pyd.Normal(mean_iq,10))
      return the_iq.item()

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
      loca = pyro.sample(self.namegen("location_"),pyd.Bernoulli(euro_prob))
      location = 'eu' if loca.item() == 1.0 else 'am'
      return location

  def gen_publications(self):
      multiplier = pyro.sample(self.namegen("pub_multiplier"),pyd.Poisson(3)).item()
      return  pyro.sample(self.namegen("number_publication"),pyd.Poisson((multiplier+1)*self.iq/100.)).item()

  def gen_citations(self):
      h_index_computed = pyro.sample(self.namegen("h_index"),pyd.Uniform(1,10))
      h_index = h_index_computed.item()
      if h_index < 0:
        h_index = 0
      return int(h_index*self.publications)


  def goal_generator(self): 
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


  def gen_reputation_points(self,model):
      '''
      Reputation points are computed every tick and weighted sum of your citations, research experience and imperfectly your IQ
      '''
      w_citations = 1.5
      w_publications = 1.2
      w_iq = pyro.sample(self.namegen("how_iq_matters"),pyd.Uniform(0.2,0.45)).item()

      agent_list = model.schedule.agents
      cit_list = [agent.citations for agent in agent_list if agent.category == 'J']
      pub_list = [agent.publications for agent in agent_list if agent.category == 'J']
      iq_list  = [agent.iq for agent in agent_list if agent.category == 'J']

      rep_points = (w_citations*(self.citations/max(cit_list))+ w_publications*(self.publications/max(pub_list)) + w_iq*(self.iq/max(iq_list)))/(w_citations+w_publications+w_iq)
      return rep_points

  def compute_reputation(self,model):
      agent_list = model.schedule.agents
      rep_list = []
      for agent in agent_list:
        if agent.category == 'J':
          rep_list.append(agent.reputation_points)
      sorted_rep = sorted(rep_list)
      return sorted_rep.index(self.reputation_points)

  def selelct_topic(self):
      k = self.numtopics = 5
      return pyro.sample(self.namegen("topic_select"),pyd.Categorical(torch.tensor([ 1/k ]*k))).item()


  def step_stage_1(self):
      self.reputation_points = self.gen_reputation_points(self.model)

  def step_stage_2(self):
      self.reputation = self.compute_reputation(self.model)

  def step_stage_final(self):
      print(self.unique_id,"has a reputation of ",self.reputation,"because thier points are",self.reputation_points, "and citations are",self.citations)



## Todo : Add Salary and money component
class Senior(Agent):
  """An agent who is on the Episthemic Landscape makes bids on Funding Proposals"""
  def __init__(self,unique_id,model,affliation,promoted_attrs = None):
      super().__init__(unique_id,model)
      self.unique_id = 'S_' + str(len([agent for agent in model.schedule.agents if agent.category == 'S']) + 1)
      self.category = 'S'   # Senior Researcher Category
      self.numtopics = 5
      self.topic_interested = self.select_topic()                   # We assume that there are 5 topics
      self.ambitions = self.goal_generator()
      self.affliation = affliation
      self.landscape = self.set_epithemic_landscape()
      self.vision = 3
      self.pos_x = np.random.randint(0,self.model.elsize)
      self.pos_y = np.random.randint(0,self.model.elsize)
      
      if promoted_attrs:
        self.iq = promoted_attrs['iq']
        self.location = promoted_attrs['location']
        self.publications =  promoted_attrs['publications']
        self.citations = promoted_attrs['citations']
        self.ambitions = promoted_attrs['ambitions']
        self.reputation_points = promoted_attrs['reputation_points']
        #self.reputation = self.compute_reputation()
      else:
        self.iq = self.iq_gen()
        self.location = self.location_generator()
        self.publications = self.gen_publications()
        self.citations = self.gen_citations()
        self.ambitions = self.goal_generator()
        self.reputation_points = 0
        

  def namegen(self,prefix):
      return prefix + "_" + str(self.unique_id)
  
  def iq_gen(self):
      '''
      Defines the IQ of the agent, this IQ is sampled from two gaussians G0 and G1
      here G0 = N(95,10) and G1 = N(115,10) with a joint Bernolli distribution of 0.95 for Junior Researchers
      '''
      will_research = pyro.sample("will_research",pyd.Bernoulli(0.999))
      mean_iq = list([100,125])[int(will_research.item())]
      the_iq = pyro.sample("the_iq",pyd.Normal(mean_iq,10))
      return the_iq.item()

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
      loca = pyro.sample(self.namegen("location_"),pyd.Bernoulli(euro_prob))
      location = 'eu' if loca.item() == 1.0 else 'am'
      return location

  def set_epithemic_landscape(self):
      return [agent for agent in self.model.schedule.agents if (agent.category == 'Elandscape' and 
                                                                agent.topic == self.topic_interested)][0]     # Because the array will only have one element which is THE Landscape


  def gen_publications(self):
      multiplier = pyro.sample(self.namegen("pub_multiplier"),pyd.Poisson(15)).item()
      return  pyro.sample(self.namegen("number_publication"),pyd.Poisson((multiplier+1)*self.iq/100.)).item()

  def gen_citations(self):
      h_index_computed = pyro.sample(self.namegen("h_index"),pyd.Uniform(1,10))
      h_index = h_index_computed.item()
      if h_index < 0:
        h_index = 0
      return int(h_index*self.publications)


  def l2_dist(self,x,y):
      if x == self.pos_x and y == self.pos_y:
        return 1
      dis = np.sqrt(np.abs((self.pos_y-x)**2 - (self.pos_x-y)**2))
      if dis == 0:
        return 1
      return dis

  def make_funding_bid(self):
      x_arr = []
      y_arr = []

      vision = self.vision
      max_size = self.model.elsize
      w = self.ambitions
      for xi in range(-vision , vision + 1):
        for yi in range(-vision,vision + 1):
          if (xi**2 + yi**2 <= (vision+0.5)**2):
            x_arr.append((self.pos_x+xi)%max_size)
            y_arr.append((self.pos_y+yi)%max_size)

      visible_x = y_arr    # The switch in x and y is crutial, DO NOT CHANGE THIS
      visible_y = x_arr 
      temp_mat = np.zeros([max_size,max_size])
      for x,y in zip(visible_x,visible_y):
        temp_mat[x][y] = (w[2]*self.landscape.matrix[x][y] + w[1]*self.landscape.diff_matrix[x][y])/self.l2_dist(x,y)
        

      #print(point_x,point_y)
      #print(visible_x)
      #print(visible_y)
      #max_index = np.unravel_index(temp_mat.argmax(), temp_mat.shape)
      max_index = np.unravel_index(temp_mat.argmax(), temp_mat.shape)
      self.pos_x = max_index[1]
      self.pos_y = max_index[0]
      self.current_bid = self.landscape.matrix[self.pos_x,self.pos_y]
      print(self.current_bid,"is the bid for",self.unique_id)

  def goal_generator(self): 
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


  def gen_reputation_points(self,model):
      '''
      Reputation points are computed every tick and weighted sum of your citations, research experience and imperfectly your IQ
      '''
      w_citations = 2
      w_publications = 1.1
      w_iq = pyro.sample(self.namegen("how_iq_matters"),pyd.Uniform(0.1,0.15)).item()

      agent_list = model.schedule.agents
      cit_list = [agent.citations for agent in agent_list if agent.category == 'S']
      pub_list = [agent.publications for agent in agent_list if agent.category == 'S']
      iq_list  = [agent.iq for agent in agent_list if agent.category == 'S']

      rep_points = (w_citations*(self.citations/max(cit_list))+ w_publications*(self.publications/max(pub_list)) + 
                    w_iq*(self.iq/max(iq_list)))/(w_citations+w_publications+w_iq)
      return rep_points

  def compute_reputation(self,model):
      agent_list = model.schedule.agents
      rep_list = []
      for agent in agent_list:
        if agent.category == 'S':
          rep_list.append(agent.reputation_points)
      sorted_rep = sorted(rep_list)
      #print(sorted_rep)
      return sorted_rep.index(self.reputation_points)

  def select_topic(self):
      k = self.numtopics
      return pyro.sample(self.namegen("topic_select"),pyd.Categorical(torch.tensor([ 1/k ]*k))).item()


  def step_stage_1(self):
      
      self.reputation_points = self.gen_reputation_points(self.model)
      print(self.pos_x,self.pos_y)

  def step_stage_2(self):
      #print(self.unique_id,"the interest is in",self.topic_interested)
      self.reputation = self.compute_reputation(self.model)
      self.make_funding_bid()
      

  def printing_step(self):
      #print(self.unique_id,"has employment of",self.employed,"and an affiliation of",self.affliation)
      print(self.unique_id,"has a reputation of ",self.reputation, ",cits are",self.citations,
            "and pubs are",self.publications,"and an affiliation of",self.affliation)
      #print("=======")

  def step_stage_3(self):
      pass

  def step_stage_final(self):
      #print(self.unique_id,"has a reputation of ",self.reputation,"because thier points are",self.reputation_points, ",citations are",self.citations,"and publications are",self.publications)
      pass