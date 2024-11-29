# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:30:22 2024

@author: lshen
"""
import numpy as np
from PDcode import *
from KHomology import KhovanovHomology_tangle
import time

def explain_result(result):
    """
    Explain the information in a dictionary where the keys represent dimensions
    and the values are lists of quantum values.
    
    Parameters:
        dictionary (dict): The dictionary with keys as dimensions and values as lists of quantum values.
        
    Prints:
        A statement for each key-value pair explaining the homology class information.
    """
    for dim, values in result.items():
        if not values:
            pass
        else:
            for value in values:
                print(f"Detect a homology class of dimension {dim} with quantum degree {value}.")
                
                
def calculate_Khovanov_homology(curves,direction=[0,0,1]):
    def height_shift(generators, a):
        return {k + a: v for k, v in generators.items()}

    def degree_shift(generators, a):
        return {k: [x + a for x in v] for k, v in generators.items()}
    
    def tensor_arc(generators):
        return {k: [x-1 for x in v] for k, v in generators.items()}
    def tensor_circle(generators):
        return {k:[x-1 for x in v].append([x+1 for x in v]) for k,v in generators.items()}

    result=[]

    start_time = time.time()
    
    tangle,Crossings = get_tangle(curves,direction)
    #plot_curves([curve.projected_points for curve in tangle],title=i)
    pd = PDcode(tangle,simplified=True)
    pdcode= pd.get_pdcode()
    print(pdcode)
    
    
    qarc = pd.qarc_count
    qcircle=pd.qcircle_count
    qpositive = pd.positive_crossings
    qnegative = pd.negative_crossings
    
    
    if pdcode:
        KHT = KhovanovHomology_tangle(pdcode)
        quantumed_generators = KHT.quantumed_generators
    else:
        quantumed_generators={0:[0]}
    #height_shift:
    quantumed_generators = height_shift(quantumed_generators, -qnegative+qpositive)
    #degree shift
    quantumed_generators = degree_shift(quantumed_generators,-qnegative+qpositive)
    for i in range(qarc):
        quantumed_generators = tensor_arc(quantumed_generators)
    for i in range(qcircle):
        quantumed_generators = tensor_circle(quantumed_generators)
    #print("arc/circle:",qarc,qcircle)
    
    
    end_time = time.time()
    #print('time:{}'.format(-start_time+end_time))
    
    khovanov_homology_class= quantumed_generators
    return khovanov_homology_class
#read data

component1 = [(-0.4, 0.0, 0.0), (2.4, 0.0, 0.0)]
component2 = [(2.2, -0.34641016151377546, 1.0), (0.8, 2.0784609690826525, 1.0)]
component3 = [(1.2, 2.0784609690826525, 2.0), (-0.2, -0.34641016151377546, 2.0)]
component = [component1,component2,component3]

result = calculate_Khovanov_homology(component)
explain_result(result)


