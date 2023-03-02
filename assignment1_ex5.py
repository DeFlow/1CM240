# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:10:34 2023

@author: 20192900
"""

import numpy as np, numpy.random
import numpy as np
import scipy 

states1= np.array([i for i in range(0,11)])
states2= np.array([i for i in range(0,11)])
states3= np.array([i for i in range(0,11)])
print(states1)
actions1= np.array([i for i in range(0,11)])
actions2= np.array([i for i in range(0,11)])
actions3= np.array([i for i in range(0,11)])

h=1  #holding costs
p=19  # lost sales costs
I_1= 5
I_2= 8
I_3= 6

pi_1 = np.array([max(I_1-s,0) for s in states1])
pi_2 = np.array([max(I_2-s,0) for s in states2])
pi_3 = np.array([max(I_3-s,0) for s in states3])
print(pi_1)
V_1 = np.array([0.0 for s in states1])
V_2 = np.array([0.0 for s in states1])
V_3 = np.array([0.0 for s in states1])

demanddict1={1:0.1, 2:0.1, 3:0.1, 4:0.1, 5:0.1, 6:0.1, 7:0.1, 8:0.1, 9:0.1, 10:0.1}
demanddict2={1:0.1, 2:0.1, 3:0.1, 4:0.1, 5:0.1, 6:0.1, 7:0.1, 8:0.1, 9:0.1, 10:0.1}
demanddict3={1:0.1, 2:0.1, 3:0.1, 4:0.1, 5:0.1, 6:0.1, 7:0.1, 8:0.1, 9:0.1, 10:0.1}

discount= 0.8
while True:
        
        max_diff_1 = 0  # Initialize max difference
        max_diff_2 = 0
        max_diff_3 = 0
        
        for s in states1:
            
           
            # we order pi[s], so our inventory level jumps to = s + pi[s].
            # store this in temp_inv_level
            temp_inv_level_1 = s+pi_1[s]
            temp_inv_level_2 = s+pi_2[s]
            temp_inv_level_3 = s+pi_3[s]
            
            val_1 = 0
            val_2 = 0
            val_3 = 0                 
            for demand in demanddict1:
               
                # find the probability of 'demand'
                # determine the resulting inventory level
                # calculat the cost of having that inventory level
                # update value:
                    
                prob_1 = demanddict1[demand]
                prob_2 = demanddict2[demand]
                prob_3 = demanddict3[demand]
                
                inv_level_1 = temp_inv_level_1-demand
                inv_level_2 = temp_inv_level_2-demand
                inv_level_3 = temp_inv_level_3-demand
                
                direct_reward_1 = -h*max(0,inv_level_1-demand)+p*min(0,inv_level_1-demand)
                direct_reward_2 = -h*max(0,inv_level_2-demand)+p*min(0,inv_level_2-demand)
                direct_reward_3 = -h*max(0,inv_level_3-demand)+p*min(0,inv_level_3-demand)
                
                val_1 += prob_1 * (direct_reward_1 + discount * V_1[inv_level_1])
                val_2 += prob_2 * (direct_reward_2 + discount * V_2[inv_level_2])
                val_3 += prob_3 * (direct_reward_3 + discount * V_3[inv_level_3])
                
            
            print("Value of state for store 1: ", val_1)
            print("Value of state for store 2: ", val_2)
            print("Value of state for store 3: ", val_3)
            
            # Update maximum difference
            max_diff_1 = max(max_diff_1, abs(val_1 - V_1[s]))
            max_diff_2 = max(max_diff_2, abs(val_2 - V_2[s]))
            max_diff_3 = max(max_diff_3, abs(val_3 - V_3[s]))
            V_1[s] = val_1  # Update V
            V_2[s] = val_2
            V_3[s] = val_3
            
        # If diff smaller than threshold delta for all states, algorithm terminates
        if max_diff_1 < 0.01 and max_diff_3 < 0.01 and max_diff_3 < 0.01:
            break
