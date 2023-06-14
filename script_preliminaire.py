# -*- coding: utf-8 -*-
"""
Created on Thu May 18 05:05:39 2023

@author: user
"""

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
import sklearn.metrics as metrics

import lib.gabyte as gabyte


# Fonction objective pour GA
def obj_fct( values,
             Predictors,
             classes,
             groups,
             score_method="f1",
             correction_method="",
             lbd=0):
    
    
    # Data selection (CT textures)
    # values[0]: CT or CECT
    # values[1]: Algorithme de discrétisation
    # values[2]: Includes CT or CECT
    # values[8->8+nb_variables]: Feature selections
    X = Predictors["CT_CECT"][int(values[0])][int(values[1])]
    X_s = X.shape[1]
    if( int(values[2]) ):
        X[:,values[8:8+X_s]==1]
        
    # Data selection (clinical)
    # values[3]: Includes clinical data
    # values[y]: Variable selection
    if( values[3]  ):
        Xt = Predictors["clinical"]
        Xt_s = Xt.shape[1]
        X = np.append(X,Xt[:,values[8+X_s:8+X_s+Xt_s]==1],axis=1)
    
    
    # If at least one predictor
    if X.shape[1]>0:
    
        # Initialize score
        score = 0
        
        # Cross-validation | For each subgroup
        for group_id in range(int(np.max(groups))):
        
            # Create model with hyperparameters
            model = XGBClassifier(  
                                    eta=        values[4],
                                    gamma=      1e-15+(values[5]/(1-values[5]+1e-15)), # Transform to 0 to infinity
                                    max_depth=  int(values[6]),
                                    subsample=  values[7]
                                  )
            
            # Fit model
            model.fit( X[groups != group_id,:] , classes[groups != group_id] )
            probs = model.predict_proba(X[groups==group_id,:])[:, 1]
            predictions = model.predict(X[groups==group_id,:])
            
            
            # Calculate score
            if score_method=="f1":
                score += metrics.f1_score(
                                            classes[groups==group_id], 
                                            predictions
                                         )
            if score_method=="accuracy":
                score += metrics.accuracy(
                                            classes[groups==group_id],
                                            predictions
                                         )
            if score_method=="auc":
                score += metrics.roc_auc_score(
                                            classes[groups==group_id],
                                            probs
                                         )
          
        score = score/(np.max(groups)+1)
            
        # Score correction
        if(correction_method == "bias"):
            score = score - lbd*X.shape[1]
        elif(correction_method == "division"):
            score = score / (X.shape[1]**(lbd))
    
        return -score
    
    else:
        return np.inf
            
        




class ga_optim:
    
    def __init__(self,
                 Predictors,
                 classes,
                 nb_groups=5,
                 score_method="f1",
                 correction_method="",
                 lbd=1):
        
        # Save constant inputs
        constants =       {
                            "Predictors":Predictors,
                            "classes":classes,
                            "score_method":score_method,
                            "correction_method":correction_method,
                            "lbd":lbd
                          }
        
        # Extract number of columns in predictors
        nb_columns_textures = Predictors["CT_CECT"][0][0].shape[1]
        nb_columns_clinical = Predictors["clinical"].shape[1]
        total_columns = nb_columns_textures + nb_columns_clinical
        
        # Create balanced groups
        self.groups = np.array([np.inf]*len(classes))
        position_array = np.arange(len(classes))
        for class_nb in [0,1]: # For each class number
            positions_class = position_array[classes==class_nb]
            np.random.shuffle(positions_class) # Shuffle class related positions
            gdivision = len(positions_class)/(nb_groups)
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
         [  0,              11,             4, ],                           # 1 Algorithme de discrétisation
         [  0,              1,              1, ],                           # 2 Use CE/CECT
         [  0,              1,              1, ],                           # 3 Use clinical data
         [  0.01,           0.2,            8, ],                           # 4 eta
         [  0,              1,              16,],                           # 5 gamma
         [  3,              10,             3, ],                           # 6 Max depth
         [  0.0001,         0.9999,         8, ],                           # 7 Subsample
        ])
        for _ in range(total_columns):
            variable_definitions = np.append(variable_definitions,
                                             [[  0,  1,  1, ]],
                                             axis=0)
        
        a = 0
        # Initialize GA optimizer
        properties = {
                "min_val":  variable_definitions[:,0],
                "max_val":  variable_definitions[:,1],
                "nb_bytes": variable_definitions[:,2],
            }
        self.ga_model = gabyte.ga(obj_fct,
                                  properties,
                                  25,
                                  constant_arguments=constants)
        
    
    def iterate(self,nb_iterations=100):
        self.ga_model.iterate(nb_iterations)
        
        
        

if __name__=="__main__":
    
    Predictors={}
    Predictors["CT_CECT"] = []
    for i in range(2):
        x = []
        for j in range(12):
            x.append( np.random.randn(500,25) )
        Predictors["CT_CECT"].append(x)
    Predictors["clinical"] = np.random.randn(500,10)

    classes = np.random.randint(0,2,[500,])
    
    optimizer = ga_optim(Predictors,classes,100,"f1","bias",0.05)
    
    optimizer.iterate(100)
    
    