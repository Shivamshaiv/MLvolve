class Labs(Agent):
  """An agent who is a junior researcher PhD, postdoc , early Researchers"""
  def __init__(self,unique_id,model,promoted_attrs = None):
      super().__init__(unique_id,model)
      self.unique_id = 'Lab_' + str(len([agent for agent in model.schedule.agents]) + 1)
      self.category = self.give_category()
      self.numtopics = 5
      self.lab_repute = self.gen_lab_repute()


  def namegen(self,prefix):
      return prefix + "_" + str(self.unique_id)

  def give_category(self):
      return "lab"

  def compute_capacity_j(self):
      capacity_j = pyro.sample(self.namegen("capcity_j"),pyd.Poisson(2)).item() + 1
      return capacity_j

  def gen_lab_repute(self):
      repute = pyro.sample(self.namegen("repute_l"),pyd.Normal(0,1)).item()
      return repute

  def print_agent_list(self,agent_list):
      list_name = [agent.unique_id for agent in agent_list]
      print(list_name)

  def recruit_juniors(self):
      other_labs =     [agent for agent in self.model.schedule.agents if (agent.category == 'lab')]
      sorted_labs = sorted(other_labs, key=lambda x: x.lab_repute, reverse=True)
      sorted_lab_capacity = [agent.capacity_j_initial for agent in sorted_labs]
      own_position = sorted_labs.index(self)
      self.lab_rank = own_position
      juniors_agents = [agent for agent in self.model.schedule.agents if (agent.category == 'J')]
      
      sorted_juniors = sorted(juniors_agents, key=lambda x: x.reputation, reverse=True)
      #self.print_agent_list(sorted_juniors)
      #self.print_agent_list(sorted_labs)
      sum_till = int(sum(sorted_lab_capacity[:own_position]))
      print(sum_till)
      if len(juniors_agents) < sum_till:
          return
      for agent in sorted_juniors[sum_till:]:
          if self.capacity_j > 0:
            agent.employed = True
            agent.affliation = self.unique_id
            self.capacity_j -= 1
    
  def step_stage_1(self):
      #print("Reputation of lab is",self.lab_repute)
      self.capacity_j = self.compute_capacity_j()
      self.capacity_j_initial = self.capacity_j + 0
      #self.lab_repute = self.gen_lab_repute()

  def step_stage_2(self):
      pass

  def step_stage_3(self):
      self.recruit_juniors()
      

  def step_stage_final(self):
      print("Reputation Rank of lab is",self.lab_rank,"Vacancy remaining in",self.unique_id,"is of",self.capacity_j,"people out of actual positions",self.capacity_j_initial)

class WorldModel(Model):
    """A model with some number of agents."""
    def __init__(self, N,num_labs):
        self.num_agents = N
        self.num_labs = num_labs
        model_stages = ["step_stage_1", "step_stage_2","step_stage_3","step_stage_final"]
        self.schedule = StagedActivation(self,model_stages)
        lab_arr = []
        # Create agents
        for i in range(self.num_agents):
            #a = Student(0,self)
            #self.schedule.add(a)
            b = Junior(i,self)
            self.schedule.add(b)
        for j in range(self.num_labs):
            c = Labs(j,self)
            lab_arr.append(c)
        #self.sorted_labs = sorted(lab_arr, key=lambda x: x.lab_repute, reverse=True)
        #print("The lenght of",len(sorted_labs))
        #for lab in self.sorted_labs:
            self.schedule.add(c)
            #print("Lab",lab,"is added")
        
        

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()