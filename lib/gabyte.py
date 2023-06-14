# -*- coding: utf-8 -*-




import numpy as np
from tqdm import tqdm





#%% Byte conversion
    
def byte_conversion(x,properties):

  count = 0
  conversion_final = np.zeros(len(properties["nb_bytes"]))

  for i in range(len(properties["nb_bytes"])):
    selection = x[count:count+properties["nb_bytes"][i]]

    b = np.array(selection,dtype=str).tolist()
    stringbyte = "".join(b)
    conversion = int(stringbyte, 2)/(2**len(stringbyte)-1)
    conversion_final[i] = properties["min_val"][i] + \
      (properties["max_val"][i]-properties["min_val"][i])*conversion

    count += properties["nb_bytes"][i]

  return conversion_final



#%% GA functions

# OBJECTIVE : MAXIMIZE
def cal_pop_fitness(pop,properties,obj_fct,constant_arguments):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    
    fitness = np.zeros(len(pop))

    for i in range(len(pop)):
        
      # Convert bytes to deltas
      x = pop[i]
      vals = byte_conversion(x,properties)
      fitness[i] = -obj_fct(vals,**constant_arguments)

    return fitness


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)

    for k in range(offspring_size[0]):
        
        # Random permutation of parents
        par_perm = np.random.permutation(len(parents))
        
        # Index of the first parent to mate.
        parent1_idx = par_perm[0]
        # Index of the second parent to mate.
        parent2_idx = par_perm[1]
        
        # Random selection of genes
        h = np.random.permutation(len(parents.T))
        pos1 = h[:int(len(parents.T)/2)]
        pos2 = h[int(len(parents.T)/2):]

        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, pos1] = parents[parent1_idx, pos1]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, pos2] = parents[parent2_idx, pos2]

    return offspring


def mutation(offspring_crossover,prop_mut):
    # Mutation changes a single gene in each offspring randomly.
    nb_mut = int( np.sum(offspring_crossover*0+1)*prop_mut )
    for _ in range(nb_mut):
        # The random value to be added to the gene.
        rd1 = np.random.randint(0,len(offspring_crossover))
        rd2 = np.random.randint(0,len(offspring_crossover.T))
        offspring_crossover[rd1, rd2] = np.abs(offspring_crossover[rd1, rd2] - 1)

    return offspring_crossover



#%% GA object


class ga:
    
    
    
    def __init__(self,obj_fct,properties,sol_per_pop,ratio_new=.9,\
                 prop_mut=.1,new_population=None,constant_arguments={}):
    
        self.history = []
        
        self.obj_fct = obj_fct
        self.properties = properties
        self.properties["nb_bytes"] = np.array(self.properties["nb_bytes"], dtype=int)
        self.properties["min_val"]  = np.array(self.properties["min_val"], dtype=float)
        self.properties["max_val"]  = np.array(self.properties["max_val"], dtype=float)
        self.constant_arguments = constant_arguments
                    
        # Paramètres ajustables arbitraires
        self.num_parents_mating = int(sol_per_pop - ratio_new*sol_per_pop)
        self.prop_mut = prop_mut
        
        # Defining the population size
        self.pop_size = (sol_per_pop,np.sum(properties["nb_bytes"]))
        
        #Creating the initial population.
        if( new_population ):
            0
        else:
            new_population = np.random.randint(0,2, size=self.pop_size)
            
        self.new_population = new_population
        
        
        
    def single_textures(self):
        """
        Retire l'attribution d'un feature à multiples algorithmes de discrétisation

        Returns
        -------
        None.

        """
        
        pos_discretization = np.arange(self.nb_columns_discretization)
        pos_texture = np.arange(self.nb_columns_textures)
        
        for i in range(len(self.new_population)):
            
            sel = self.new_population[i,\
                    -self.total_columns:-self.total_columns+\
                     self.nb_columns_textures*self.nb_columns_discretization]
            sel = sel.reshape([self.nb_columns_discretization,
                               self.nb_columns_textures])
            sel_s = np.sum(sel,axis=0)
            
            for pos_t in pos_texture[ sel_s>1 ]:
                pos_d = pos_discretization[ sel[:,pos_t]==1 ]
                np.random.shuffle(pos_d)
                sel[:,pos_t] = 0
                sel[pos_d[0],pos_t] = 1
                
            self.new_population[i,\
            -self.total_columns:-self.total_columns+\
             self.nb_columns_textures*self.nb_columns_discretization] = sel.flatten()
                
        
        
    def maximize_nb_var(self):
        """
        Retire les variables en trop

        Returns
        -------
        None.

        """
        
        pos = np.arange(self.total_columns) # Initialize array
        
        for i in range(len(self.new_population)):
            sel = self.new_population[i,-self.total_columns:]
            if( np.sum(sel)>self.max_var ):
                sel[ : ] = 0
                np.random.shuffle( pos ) # Shuffle pos
                sel[ pos[:self.max_var] ] = 1 
                self.new_population[i,-self.total_columns:] = sel
        
        
        
    def iterate(self,num_generations,verbose=True):
        
        self.single_textures()
        self.maximize_nb_var()
        
        if(verbose):
            pbar = tqdm(total=num_generations, position=0, leave=True)
        
        for generation in range(num_generations):

            
            # Measuring the fitness of each chromosome in the population.
            if generation == 0:
                # Calculate all fitness
                fitness = cal_pop_fitness( self.new_population, self.properties, 
                                          self.obj_fct, self.constant_arguments )
            else: # Parent fitness is already known
                fitness_children = cal_pop_fitness( self.new_population[self.num_parents_mating:], 
                        self.properties, self.obj_fct, self.constant_arguments )
                fitness = np.append(fitness_parents,fitness_children)
            
            # Save to history
            self.history.append( [ self.new_population.copy() , fitness.copy() ] )
                
            # Selecting the best parents in the population for mating.
            parents = self.new_population[ \
                            np.argsort(-fitness)[:self.num_parents_mating] ]
            fitness_parents = fitness[ np.argsort(-fitness)[:self.num_parents_mating] ]
            
            # Generating next generation using crossover.
            offspring_crossover = crossover(parents,
                            offspring_size=(self.pop_size[0]-parents.shape[0],
                                        np.sum(self.properties["nb_bytes"])))
            
            # Adding some variations to the offsrping using mutation.
            offspring_mutation = mutation(offspring_crossover,self.prop_mut)
            
            # Creating the new population based on the parents and offspring.
            self.new_population[:parents.shape[0], :] = parents
            self.new_population[parents.shape[0]:, :] = offspring_mutation

            self.single_textures()
            self.maximize_nb_var()

            if(verbose):
                pbar.update(1)
                pbar.set_description("Loss = {}".format(-np.max(fitness)))
        


    def get_solution(self):
        
        fitness = cal_pop_fitness(    self.new_population, \
                                      self.properties, 
                                      self.obj_fct, 
                                      self.constant_arguments )
        best_pos = np.argsort(-fitness)[0]
        
        sol = self.new_population[best_pos]
        sol = byte_conversion(sol,self.properties)

        return sol
        
        