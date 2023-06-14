# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 07:35:54 2023

@author: user
"""


import numpy as np

# Simulate predictors
Predictors={}
Predictors["CT_CECT"] = []
for i in range(2):
    x = []
    for j in range(12):
        x.append( np.random.randn(500,25) )
    Predictors["CT_CECT"].append(x)
Predictors["clinical"] = np.random.randn(500,10)

Predictors["CT_CECT_morph"] = [np.random.randn(500,10),
                               np.random.randn(500,10)]
Predictors["CT_CECT_intensity"] = [np.random.randn(500,15),
                                   np.random.randn(500,15)]


# Simulate classes (variable to predict)
classes = np.random.randint(0,2,[500,])

