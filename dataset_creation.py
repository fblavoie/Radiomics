# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:08:24 2023

@author: user
"""



import pandas as pd
import numpy as np


radiomic_variables = open("data/radiomics_variables.txt","r").read().split("||")[1:-1]


# Get all discretization algorithms
all_variants = []
for i in range(79,len(radiomic_variables)):
    divisions = radiomic_variables[i].split("_")
    all_variants.append(divisions[-2] + "_" + divisions[-1])
all_variants = np.unique(all_variants)


# Get common individual IDs
l1 = list(np.array(pd.read_csv("data/Results_radiomics_CT_image.csv"))[:,0])
l2 = list(np.array(pd.read_csv("data/Results_radiomics_CECT_image.csv"))[:,0])
l3 = list(np.array(pd.read_csv("data/Données_Clinique__CSV.csv"))[:,0])
l4 = list(np.array(pd.read_csv("data/Outcome_4.csv"))[:,0])
all_commons = []
for pid in l1:
    if (pid in l2) and (pid in l3) and (pid in l4): # If IDs are in all datasets
        all_commons.append( pid )


# Create dataset 
Predictors = {"CT_CECT":[], "CT_CECT_morph":[], "CT_CECT_intensity":[]}

for dataset in ["CT","CECT"]: # 

    # Load radiomic data
    X = np.array(pd.read_csv(f"data/Results_radiomics_{dataset}_image.csv"))
    patient_id = X[:,0]
    X = X[:,1:]
    
    # Matrix filtering with common IDs
    pos_array = []
    for pid in all_commons:
        pos_array.append ( np.arange(len(X))[ patient_id == pid ][0] )
    pos_array = np.array(pos_array)
    X = X[pos_array]
    
    # Create the 2D matrices
    X_morphological = X[:,:29]
    X_intensity = X[:,29:79]
    
    # Create the 3D matrix
    all_X = []
    for i in all_variants:
        all_X.append( X[:,0:0] )
    for i in range(79,len(radiomic_variables)):
        divisions = radiomic_variables[i].split("_")
        variant = divisions[-2] + "_" + divisions[-1]
        pos = np.arange(len(all_variants))[all_variants==variant][0]
        all_X[pos] = np.append(all_X[pos],X[:,i:i+1],axis=1)
    
    # Insert data into the predictors 
    Predictors["CT_CECT"].append(np.array(all_X))
    Predictors["CT_CECT_morph"].append(X_morphological)
    Predictors["CT_CECT_intensity"].append(X_intensity)
    

# Create the clinical matrix
pos_array = []
X = pd.read_excel("data/Données_Clinique__CSV.xlsx", 
                   dtype= { "GENDER":"category",
                            "ETHNICITY":"category",
                            "FAMILY_HISTORY":"category",
                            "SMOKING_STATUS":"category",
                            "LESION_SIDE_IMAGING":"category",
                            "LESION_POLE_IMAGING":"category",
                            "PHYTIC_TYPE_IMAGING":"category"})
patient_id = np.array(X.iloc[:,0])
X = X.iloc[:,1:] # Remove first column
for pid in all_commons:
    pos_array.append ( np.arange(len(X))[ patient_id == pid ][0] )
pos_array = np.array(pos_array)
X = X.iloc[pos_array]
Predictors["clinical"] = X


# Create the y (classes) matrix
classes = pd.read_csv("data/Outcome_4.csv")
classes = np.array(classes[pd.notnull(classes["PATHOLOGY_GRADE"])])
patient_id = classes[:,0]
classes = classes[:,1:]
pos_array = []
for pid in all_commons:
    pos_array.append ( np.arange(len(classes))[ patient_id == pid ][0] )
pos_array = np.array(pos_array)
classes = np.array(classes[pos_array],dtype=int).flatten()



