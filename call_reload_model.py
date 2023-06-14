# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:35:34 2023

@author: user
"""



folder = "2023_06_06_11_32"


import pickle

from fct_ga import ga_optim

model = f"models/{folder}/model.pkl"
optimizer = pickle.load( open(model,"rb") )

