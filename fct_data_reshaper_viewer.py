# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:33:16 2023

@author: user
"""


from tabulate import tabulate



def data_reshaper_viewer(sol):

    print("")
    

    # XgBoost Parameters
    print("")
    print("XGBoost parameter table")
    table =  [ 
                [ "eta",         sol[6] ],
                [ "gamma",       sol[7] ],
                [ "Max depth",   sol[8] ],
                [ "Subsample",   sol[9] ],
             ]
    print(tabulate(table))
    

    # CE / CECT
    if sol[0] == 0:
        print("Utilisation de CT")
    elif sol[0] == 1:
        print("Utilisation de CECT")
    
