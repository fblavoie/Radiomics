# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 07:14:10 2023

@author: user
"""

from xgboost import XGBClassifier
import sklearn.metrics as metrics

import numpy as np
import pandas as pd



VAR_BASIS = 10


# Fonction objective pour GA
def obj_fct( values,
             Predictors,
             classes,
             groups,
             score_method="f1",
             correction_method="",
             lbd=0):
    
    
    # Data selection (images)
    # values[0]: CT or CECT 
    # values[1]: Algorithme de discrétisation
    
    # values[2]: Use of textural data
    # values[10->10+nb_variables]: Feature selections
    X = Predictors["CT_CECT"][int(values[0])]
    X_s = X.shape[1]
    if( (int(values[2])==1)*0==0 ): # X is included
        X = X[:,values[VAR_BASIS:\
                   VAR_BASIS+X_s]==1]
    else:
        X = np.zeros([len(X),0]) # Create matrix with no columns
    
    # values[3]: Use of morphological data
    MORPH = Predictors["CT_CECT_morph"][int(values[0])].shape[1]
    if( (int(values[3])==1)*0==0 ):
        X = np.append(X,Predictors["CT_CECT_morph"][int(values[0])][:,VAR_BASIS+X_s:\
                                                                      VAR_BASIS+X_s+MORPH],axis=1)
    
    # values[4]: Use of intensity data 
    INTENS = Predictors["CT_CECT_intensity"][int(values[0])].shape[1]
    if( (int(values[4])==1)*0==0 ):
        X = np.append(X,Predictors["CT_CECT_intensity"][int(values[0])]\
                      [:,VAR_BASIS+X_s+MORPH:\
                         VAR_BASIS+X_s+MORPH+INTENS],axis=1)
    
    # Transform X to DataFrame
    X = pd.DataFrame(X,dtype=float)
        
    # Data selection (clinical)
    # values[5]: Includes clinical data
    # values[y]: Variable selection
    if( (int(values[5])==1)*0==0 ):
        Xt = Predictors["clinical"]
        Xt_s = Xt.shape[1]
        Xt_select = Xt.iloc[:,values[VAR_BASIS+MORPH+INTENS+X_s:\
                                    VAR_BASIS+MORPH+INTENS+X_s+Xt_s]==1]
        for c in list(Xt_select.columns): # Add columns 
            X[c] = Xt_select[c]

        
    
    # If at least one predictor
    if X.shape[1]>0:
    
        # Initialize score
        score = 0
        
        # Cross-validation | For each subgroup
        for group_id in range(int(np.max(groups))):
        
            # Create model with hyperparameters
            model = XGBClassifier(  
                                    eta=        values[6],
                                    gamma=      1e-15+(values[7]/(1-values[7]+1e-15)), # Transform to 0 to infinity
                                    max_depth=  int(values[8]),
                                    subsample=  values[9],
                                    enable_categorical=True,
                                    tree_method="gpu_hist",
                                  )
            
            # Fit model
            model.fit( X.iloc[groups != group_id,:] , classes[groups != group_id] )
            probs = model.predict_proba(X.iloc[groups==group_id,:])[:, 1]
            predictions = model.predict(X.iloc[groups==group_id,:])
            
            
            # Calculate score
            
            # Les méthodes suivantes requièrent l'utilisation des prédictions
            if score_method=="f1":
                score += metrics.f1_score(
                                            classes[groups==group_id], 
                                            predictions
                                         )
            elif score_method=="accuracy":
                score += metrics.accuracy(
                                            classes[groups==group_id],
                                            predictions
                                         )
            elif score_method=="balanced_accuracy":
                score += metrics.balanced_accuracy_score(
                                            classes[groups==group_id],
                                            predictions
                                         )
            elif score_method=="recall":
                score += metrics.recall_score(
                                            classes[groups==group_id],
                                            predictions
                                         )
                
                
            # Les méthodes suivantes requièrent l'utilisation des probabilités 
            # et non des prédictions
            elif score_method=="brier":
                score += metrics.brier_score_loss(
                                            classes[groups==group_id],
                                            probs
                                         )
            elif score_method=="log_loss":
                score += metrics.log_loss(
                                            classes[groups==group_id],
                                            probs
                                         )
            if score_method=="auc":
                score += metrics.roc_auc_score(
                                            classes[groups==group_id],
                                            probs
                                         )
          
        score = score/(np.max(groups)+1)
            
        # Score correction (for variable reduction)
        if(correction_method == "bias"):
            score = score - lbd*X.shape[1]
        elif(correction_method == "division"):
            score = score / (X.shape[1]**(lbd))
    
        return -score # Must be set to negative as GA minimizes the objective function
    
    else:
        return np.inf
            
        
 