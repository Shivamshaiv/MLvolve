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



class Student(Agent):
    """An agent who is a university student."""
    def __init__(self,unique_id,model):
        super().__init__(unique_id,model)
        self.model = model
        self.unique_id = 'U_' + str(len([agent for agent in model.schedule.agents if agent.category == 'U']) + 1)
        self.category = 'U'   # University Student Category
        self.iq = self.iq_gen()
        self.ambitions = self.goal_generator()
        self.location = self.location_generator()
        self.publications = self.gen_publications()
        self.citations = self.gen_citations()
        self.reputation_points = 0
        self.employed = False
        self.affliation = None
        self.booked = False
        self.time_since_employed = 0
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
      b = pyro.sample("second_pivot",pyd.Uniform(0,1-a.item()))
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
      #print("Pub given")
      num_pub = pyro.sample(self.namegen("number_publication"),pyd.Poisson(self.iq/100.))
      return num_pub.item()

    def gen_citations(self):
      h_index_computed = pyro.sample(self.namegen("h_index"),pyd.Normal(4,1))
      h_index = h_index_computed.item() + pyro.sample(self.namegen("h_index_adding"),pyd.Uniform(0.5,1)).item()
      if h_index < 0:
        h_index = 0
      return int(h_index*self.publications)

    def gen_reputation_points(self,model):
      '''
      Reputation points are computed every tick and weighted sum of your citations, research experience and imperfectly your IQ
      '''
      w_citations = 1.5
      w_publications = 1.2
      w_iq = pyro.sample(self.namegen("how_iq_matters"),pyd.Uniform(0.75,1.5)).item()

      agent_list = model.schedule.agents
      cit_list = [agent.citations for agent in agent_list if (agent.category == 'J' or agent.category == 'U')]
      pub_list = [agent.publications for agent in agent_list if (agent.category == 'J' or agent.category == 'U')]
      iq_list  = [agent.iq for agent in agent_list if (agent.category == 'J' or agent.category == 'U')]

      rep_points = (w_citations*(self.citations/max(cit_list))+ w_publications*(self.publications/max(pub_list)) + w_iq*(self.iq/max(iq_list)))/(w_citations+w_publications+w_iq)
      return rep_points


    def compute_reputation(self,model):
      agent_list = model.schedule.agents
      rep_list = []
      for agent in agent_list:
        if (agent.category == 'J' or agent.category == 'U'):
          rep_list.append(agent.reputation_points)
      sorted_rep = sorted(rep_list)
      return sorted_rep.index(self.reputation_points)

    def promote(self):
      promotion_dict = {'iq':self.iq , 'ambitions':self.ambitions,'location':self.location,'publications':self.publications,'citations':self.citations,"reputation_points":self.reputation_points}
      return promotion_dict

    def remove_if_useless(self):
      if self.time_since_employed >= self.model.remove_thres:
        self.model.schedule.remove(self)


    def step_stage_1(self):
      self.reputation_points = self.gen_reputation_points(self.model)

    def step_stage_2(self):
      self.reputation = self.compute_reputation(self.model)

    def step_stage_3(self):
      pass


    def step_stage_4(self):
      pass

    def step_stage_5(self):
        pass

    def step_stage_6(self):
        if self.affliation is not None:
          self.time_since_employed = 0
        else:
          self.time_since_employed +=1

    def step_stage_7(self):
        pass

    def printing_step(self):
      #print(self.unique_id,"has employment of",self.employed,"and an affiliation of",self.affliation)
      print(self.unique_id,"has a reputation of ",self.reputation, ",cits are",self.citations,"and pubs are",self.publications,"and an affiliation of",self.affliation)
      #print("=======")

    def step_stage_final(self):
      #print(self.unique_id,"has a reputation of ",self.reputation,"because thier points are",self.reputation_points, ",citations are",self.citations,"and publications are",self.publications)
      self.remove_if_useless()
      self.employed = False
      self.affliation = None
      self.booked = False
      pass





class Junior(Agent):
  """An agent who is a junior researcher PhD, postdoc , early Researchers"""
  def __init__(self,unique_id,model,promoted_attrs = None):
      super().__init__(unique_id,model)
      self.unique_id = 'J_' + str(len([agent for agent in model.schedule.agents if agent.category == 'J']) + 1)
      self.category = 'J'   # Junior Researcher Category
      self.numtopics = 5
      self.topic_interested = self.select_topic()                   # We assume that there are 5 topics
      self.employed = False
      self.affliation = None
      self.time_since_employed = 0
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


  def remove_if_useless(self):
      if self.time_since_employed >= self.model.remove_thres:
        self.model.schedule.remove(self)
  
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
      cit_list = [agent.citations for agent in agent_list if (agent.category == 'J' or agent.category == 'U')]
      pub_list = [agent.publications for agent in agent_list if (agent.category == 'J' or agent.category == 'U')]
      iq_list  = [agent.iq for agent in agent_list if (agent.category == 'J' or agent.category == 'U')]

      rep_points = (w_citations*(self.citations/max(cit_list))+ w_publications*(self.publications/max(pub_list)) + w_iq*(self.iq/max(iq_list)))/(w_citations+w_publications+w_iq)
      return rep_points

  def compute_reputation(self,model):
      agent_list = model.schedule.agents
      rep_list = []
      for agent in agent_list:
        if (agent.category == 'J' or agent.category == 'U'):
          rep_list.append(agent.reputation_points)
      sorted_rep = sorted(rep_list)
      return sorted_rep.index(self.reputation_points)

  def select_topic(self):
      k = self.numtopics = 5
      return pyro.sample(self.namegen("topic_select"),pyd.Categorical(torch.tensor([ 1/k ]*k))).item()


  def step_stage_1(self):
      self.reputation_points = self.gen_reputation_points(self.model)

  def step_stage_2(self):
      self.reputation = self.compute_reputation(self.model)

  def step_stage_3(self):
      pass 

  def step_stage_4(self):
      pass

  def step_stage_5(self):
      pass

  def step_stage_6(self):
      if self.affliation is not None:
        self.time_since_employed = 0
      else:
        self.time_since_employed +=1

  def step_stage_7(self):
      pass

  def printing_step(self):
      #print(self.unique_id,"has employment of",self.employed,"and an affiliation of",self.affliation)
      print(self.unique_id,"has a reputation of ",self.reputation, ",cits are",self.citations,"and pubs are",self.publications,"and an affiliation of",self.affliation)
      #print("=======")

  def step_stage_final(self):
      self.remove_if_useless()
      self.employed = False
      self.affliation = None
      pass



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
      self.is_funded = False
      self.funding = 0
      self.funding_source = None
      
      
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
        self.lab_coupled_reputation = 0
        self.bid_value = 0
        

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
      dis = np.sqrt(np.abs((self.pos_x-x)**2 + (self.pos_y-y)**2))
      if dis == 0:
        return 1
      return dis

  def make_funding_bid(self):
      max_size = self.model.elsize
      w = self.ambitions
      if self.model.timestep > 1:
        temp_mat = np.zeros([max_size,max_size])
        for x,y in zip(self.landscape.visible_x,self.landscape.visible_y):
          temp_mat[y,x] = (w[2]*self.landscape.matrix[y,x] * w[1]*self.landscape.diff_matrix[y,x])/self.l2_dist(x,y)

        # Episilon greedy
        if pyro.sample(self.namegen("e_greedy"),pyd.Poisson(self.model.episilon)).item() == 1:
          self.pos_x = np.random.choice(self.landscape.visible_x)
          self.pos_y = np.random.choice(self.landscape.visible_y)
        else:
          max_index = np.unravel_index(temp_mat.argmax(), temp_mat.shape)
          self.pos_x = max_index[0]
          self.pos_y = max_index[1]
      self.current_bid = self.landscape.matrix[self.pos_y,self.pos_x]/self.landscape.max_height
      self.difficulty_selected = self.landscape.diff_matrix[self.pos_y,self.pos_x]
      self.bid_novelty = self.compute_novelty()
      #print(self.bid_novelty)
      w1 = 1   # How much significance matters
      w2 = 1   # How much novelty matters
      w3 = 1   # How much reputation matters
      self.bid_value = (w1*self.current_bid + w2*self.bid_novelty + w3*self.reputation_points)/(w1+w2+w3)

  def compute_novelty(self):
    if self.model.timestep <= 1:
      return 0
    else:
      invert = True
      if invert:
        discovered_inv = np.abs(1-self.landscape.bid_store)    # Inverting Discovery for novelty
      else:
        discovered_inv = self.landscape.bid_store
  
      dtimg = ndimage.distance_transform_edt(discovered_inv)
      #print(np.unique(dtimg))
      return dtimg[self.pos_y,self.pos_x]/np.max(dtimg)



  def recruit_juniors(self):
      other_seniors =     [agent for agent in self.model.schedule.agents if (agent.category == 'S' and agent.is_funded)]
      sorted_seniors = sorted(other_seniors, key=lambda x: x.lab_coupled_reputation, reverse=True)
      sorted_seniors_capacity = [agent.capacity_j_initial for agent in sorted_seniors]
      own_position = sorted_seniors.index(self)
      self.sorted_seniors = sorted_seniors  # Stored to in the variabe to remove reduntant computations
      self.senior_rank = own_position
      juniors_agents = [agent for agent in self.model.schedule.agents if (agent.category == 'J')]
      
      sorted_juniors = sorted(juniors_agents, key=lambda x: x.reputation_points, reverse=True)
      #self.print_agent_list(sorted_juniors)
      #self.print_agent_list(sorted_labs)
      sum_till = int(sum(sorted_seniors_capacity[:own_position]))
      #print(sum_till)
      if len(juniors_agents) < sum_till:
          return
      for agent in sorted_juniors[sum_till:]:
          if self.capacity_j > 0:
            agent.employed = True
            agent.affliation = self.unique_id
            self.capacity_j -= 1

  def compute_optimal_vacancies(self):
      C_ = self.landscape.C_
      m_j = self.model.m_j
      m_u = self.model.m_u
      lamb = self.model.lamb
      grant_money = self.funding
      def hiring_func(x):
        p1 = ((grant_money -x[0]*m_j - x[1]*m_u)/C_)
        p2 = x[0]
        p3 = x[1]
        if p1 < 0:
          return np.inf
        #print("(p1 + p2 + p3) is  :",(p1 + p2 + p3))
        #print("sig lamb*((p1-p2)^2:",lamb*((p1-p2)**2 + (p2-p3)**2 + (p1-p3)**2))
        return -((p1 + p2 + p3)- lamb*((p1-p2)**2 + (p2-p3)**2 + (p1-p3)**2))
      cap = [1,1]
      min_val = 10000000
      for i in range(1,11):
        for j in range(1,11):
          if i+j <= 10:
            if hiring_func([i,j]) < min_val:
              min_val = hiring_func([i,j])
              cap = [i,j]
            #print((i,j)," ",hiring_func([i,j]))
      #print(cap)
      self.capacity_j_initial = cap[0]
      #print("Capcity of",self.unique_id,"is",self.capacity_j_initial)
      self.capacity_j = cap[0]
      self.compute_productivity = ((grant_money -cap[0]*m_j - cap[1]*m_u)/C_)
      own_lab = [agent for agent in self.model.schedule.agents if (agent.category == 'lab' and agent.unique_id == self.affliation)][0]  # Only one element of array
      self.own_lab_agent = own_lab
      #print(own_lab.unique_id,"is affilicated to",self.unique_id)
      own_lab.capacity_s += cap[1]
      own_lab.capacity_s_initial += cap[1]
        
  def chance_of_project_success(self):
      d = self.difficulty_selected
      chance = 1 - np.exp(-self.project_productivity/(d))
      
      return chance

  def work_on_projects(self):
      def productivity(rep_j,rep_u,comp_prod):
        return (sum(rep_j)+sum(rep_u) + comp_prod)

      lab_students = [agent for agent in self.model.schedule.agents if (agent.category == 'U' and agent.affliation == self.affliation)]
      rep_u = []
      u_agents = []
      for agent in lab_students:
        if not agent.booked:
          if self.own_lab_agent.capacity_s_initial >= self.own_lab_agent.capacity_s:
            rep_u.append(agent.reputation_points)
            agent.booked = True
            self.own_lab_agent.capacity_s += 1
            u_agents.append(agent)


      j_agents = [agent for agent in self.model.schedule.agents if (agent.category == 'J' and agent.affliation == self.unique_id)]
      rep_j = [agent.reputation_points for agent in j_agents]
      self.project_productivity = productivity(rep_j,rep_u,self.compute_productivity)*self.reputation_points   # This includes all the agents and compute
      d = self.difficulty_selected
      chance = self.chance_of_project_success()
      self.is_successful = pyro.sample(self.namegen("Proj_success"),pyd.Bernoulli(chance)).item()
      if self.is_successful == 1:
        print("Chance of project by",self.unique_id,"is",round(chance,5),"with difficulty",round(d,3),"with bid of",round(self.bid_value,4),"topic was",self.topic_interested,".....it was a SUCCESS")
        vision = 4
        self.landscape.reduce_novelty([self.pos_y,self.pos_x],1)
        publication_generated = pyro.sample(self.namegen("publication_gen"),pyd.Poisson(self.project_productivity)).item() + 1
        citations_generated = publication_generated*pyro.sample(self.namegen("citation_gen"),pyd.Poisson(self.landscape.matrix[self.pos_y,self.pos_x])).item()  # Significance result to citations    
        self.publications+= publication_generated
        self.citations+= citations_generated
        for students in u_agents:
          students.publications+= publication_generated
          students.citations+= citations_generated
        for juniors in j_agents:
          juniors.publications+= publication_generated
          juniors.citations+= citations_generated
      else:
        print("Chance of project by",self.unique_id,"is",round(chance,5),"with difficulty",round(d,3),"with bid of",round(self.bid_value,4),"topic was",self.topic_interested,".....it was a FAILURE")
        vision = 2
        self.landscape.reduce_novelty([self.pos_y,self.pos_x],chance)

      max_size = self.model.elsize
      for xi in range(-vision , vision + 1):
        for yi in range(-vision,vision + 1):
          if (xi**2 + yi**2 <= (vision+0.5)**2):
            self.landscape.x_arr.append((self.pos_x+xi)%max_size)
            self.landscape.y_arr.append((self.pos_y+yi)%max_size)
            self.landscape.explored[(self.pos_y+yi)%max_size,(self.pos_x+xi)%max_size] = 1

      self.landscape.visible_x = self.landscape.x_arr    
      self.landscape.visible_y = self.landscape.y_arr



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
      self.make_funding_bid()
      #print(self.pos_x,self.pos_y)

  def step_stage_2(self):
      #print(self.unique_id,"the interest is in",self.topic_interested)
      self.reputation = self.compute_reputation(self.model)
      
      

  def printing_step(self):
      #print(self.unique_id,"has employment of",self.employed,"and an affiliation of",self.affliation)
      #print(self.unique_id,"has a reputation of ",self.reputation, ",cits are",self.citations,"and pubs are",self.publications,"and an affiliation of",self.affliation)
      if self.is_funded:
        print(self.unique_id,"affliated to",self.affliation, "has funding of ",self.funding,"from the source",self.funding_source, "with a bid of",self.current_bid,"and recruitment of juniors happened as",(self.capacity_j,self.capacity_j_initial))
      else:
        print(self.unique_id,"affliated to",self.affliation, "has funding of ",self.funding,"from the source",self.funding_source, "with a bid of",self.current_bid)
      #print("=======")

  def step_stage_3(self):
      if self.is_funded:
        #print("Computing vacncies for",self.unique_id)
        self.compute_optimal_vacancies()
        #print("Completed vacancies for",self.unique_id)
      


  def step_stage_4(self):
      if self.is_funded:
        #print("The capacity of",self.unique_id,"is",self.capacity_j_initial)
        self.recruit_juniors()
      

  def step_stage_5(self):
      if self.is_funded:
        self.work_on_projects()

  def step_stage_6(self):
      pass

  def step_stage_7(self):
      pass

  def step_stage_final(self):
      #print(self.unique_id,"has a reputation of ",self.reputation,"because thier points are",self.reputation_points, ",citations are",self.citations,"and publications are",self.publications)
      self.is_funded = False
      self.funding_source = None
      pass