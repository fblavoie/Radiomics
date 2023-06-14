# -*- coding: utf-8 -*-
"""
Created on Thu May 18 05:05:39 2023

@author: user
"""

import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime


from xgboost import XGBClassifier
import sklearn.metrics as metrics

import lib.gabyte as gabyte
from fct_obj import obj_fct

from tabulate import tabulate



class ga_optim:
    
    
    def __init__(self,
                 Predictors,
                 classes,
                 nb_groups=5,
                 sol_per_pop=100,
                 score_method="f1",
                 correction_method="",
                 lbd=1):
        
        # Create folder
        isExist = os.path.exists("models")
        if not isExist:
            os.mkdir("models")

        # Create model directory
        self.folder = datetime.now().strftime("%Y_%m_%d_%H_%M")
        os.mkdir("models/"+self.folder)
        
        # Save constant inputs
        constants =       {
                            "Predictors":Predictors,
                            "classes":classes,
                            "score_method":score_method,
                            "correction_method":correction_method,
                            "lbd":lbd
                          }
        
        # Archive constants
        file = open("models/"+self.folder+"/constants.pkl","wb")
        pickle.dump(constants,file)
        file.close()
        
        # Extract number of columns in predictors
        nb_columns_textures = Predictors["CT_CECT"][0][0].shape[1]
        nb_columns_discretization = Predictors["CT_CECT"][0].shape[0]
        nb_columns_morph = Predictors["CT_CECT_morph"][0].shape[1]
        nb_columns_intensity = Predictors["CT_CECT_intensity"][0].shape[1]
        nb_columns_clinical = Predictors["clinical"].shape[1]
        total_columns = nb_columns_textures*nb_columns_discretization  + \
                                                  nb_columns_clinical  + \
                                                  nb_columns_intensity + \
                                                  nb_columns_morph
        
        # Reshape texture matrices
        for i in range(2):
            X_t = Predictors["CT_CECT"][i][0]
            for j in range(1,Predictors["CT_CECT"][i].shape[0]):
                X_t = np.append(X_t,Predictors["CT_CECT"][i][j],axis=-1)
            Predictors["CT_CECT"][i] = X_t
       
        # Save number of columns
        self.nb_columns_textures = nb_columns_textures
        self.nb_columns_discretization = nb_columns_discretization
        self.nb_columns_morph = nb_columns_morph
        self.nb_columns_intensity = nb_columns_intensity
        self.nb_columns_clinical = nb_columns_clinical
        
        # Create balanced groups
        self.groups = np.array([np.inf]*len(classes))
        position_array = np.arange(len(classes))
        for class_nb in [0,1]: # For each class number
            positions_class = position_array[classes==class_nb]
            np.random.shuffle(positions_class) # Shuffle class related positions
            gdivision = len(positions_class)/nb_groups
            for group_nb in range(nb_groups):
                positions_keep = positions_class[ int(np.round(gdivision*group_nb)):\
                                                  int(np.round(gdivision*(group_nb+1))) ]
                self.groups[positions_keep] = group_nb
                
        # Add groups
        constants["groups"] = self.groups


        # Create variable definitions for GA
        variable_definitions = np.array([
         #  Min value       Max value       Nb bits
         [  0,              1,              1, ],                           # 0 CT_CECT
         [  0,              11,             4, ],                           # 1 Algorithme de discrétisation (not_used)
         [  0,              1,              1, ],                           # 2 Use CE/CECT textural data (not used)
         [  0,              1,              1, ],                           # 3 Use CE/CECT morphological data (not used)
         [  0,              1,              1, ],                           # 4 Use CE/CECT intensity data (not used)
         [  0,              1,              1, ],                           # 5 Use clinical data (not used)
         [  0.01,           0.2,            8, ],                           # 6 eta
         [  0,              1,              16,],                           # 7 gamma
         [  3,              10,             3, ],                           # 8 Max depth
         [  0.0001,         0.9999,         8, ],                           # 9 Subsample
        ])
        for _ in range(total_columns):
            variable_definitions = np.append(variable_definitions,
                                             [[  0,  1,  1, ]],
                                             axis=0)
        
        # Initialize GA optimizer
        properties = {
                "min_val":  variable_definitions[:,0],
                "max_val":  variable_definitions[:,1],
                "nb_bytes": variable_definitions[:,2],
            }
        self.ga_model = gabyte.ga(obj_fct,
                                  properties,
                                  sol_per_pop,
                                  constant_arguments=constants,
                                  prop_mut=.2)
                
        # Insert max variable arguments
        self.ga_model.total_columns = total_columns
        self.ga_model.nb_columns_textures = nb_columns_textures
        self.ga_model.nb_columns_discretization = nb_columns_discretization
        self.ga_model.max_var = 10
        
        # Initialize history
        file = open("models/"+self.folder+"/model.pkl","wb")
        pickle.dump(self,file)
        file.close()
        
        
    
    def update_model_save(self):
        
        # Initialize history
        file = open("models/"+self.folder+"/model.pkl","wb")
        pickle.dump(self,file)
        file.close()
        
    
    
    def iterate(self,nb_iterations=100):
        
        # Save model at each 10 iterations
        for i in range(int(nb_iterations//10)):
            self.ga_model.iterate(10)
            self.update_model_save()
        
        # Residual iterations
        res_iterations = int(nb_iterations%10)
        if( res_iterations>0 ):
            self.ga_model.iterate(res_iterations)
            self.update_model_save()
        
        
        
    def get_solution(self):
        sol = self.ga_model.get_solution()
        
        
        

        

# Class debug
if __name__=="__main__":
    
    from dataset_creation import Predictors, classes
    
    # Initialize the optimizer
    optimizer = ga_optim(Predictors,
                         classes,
                         nb_groups=5,
                         sol_per_pop=100,
                         score_method="log_loss",
                         )
    
    # Perform 10 GA iterations
    optimizer.iterate(100)
    
    sol = optimizer.get_solution()


