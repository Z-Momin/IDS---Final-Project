import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import gzip

path = os.getcwd()
with gzip.open(path+'\\data\\data.pkl.gzip') as dat:
    data = pickle.load(dat)
    
d = {}  
  
# Each setting corresponds to a specific action 
# Here they are labeled arbitrarily because I wasnt sure if theyre exactly right
rest = tuple([0,0,0]) 
right = tuple([1,0,0])
acc = tuple([0,1,0])
left = tuple([-1,0,0])
brake = tuple([0,0,0.2])
left_brake = tuple([-1,0,0.2])
left_acc = tuple([-1,1,0])
right_brake = tuple([1,0,0.2])
right_acc = tuple([1,1,0])

# Adding values to each key in the dictionary
d[rest] = 'Rest'
d[right] = 'Right'
d[acc] = 'Acc'
d[left] = 'Left'
d[brake] = 'Brake'
d[left_brake] = 'Left_Brake'
d[left_acc] = 'Acc_Left'
d[right_brake] = 'Right_Brake'
d[right_acc] = 'Acc_Right'

# Uncomment this if you would like to print the dictionary out

# for key, value in d.items(): 
#     print(key, ':', value)

# The compare function will return true if each element of the inputs is the same with 1e-5 tolerance
def compare(a , b):
    length = len(a)
    counter = 0 
    for i in range(length):
        if np.abs(a[i]-b[i]) < 1e-5:
            counter += 1
    if counter == length:
        return True
    else:
        return False
    
# Here discretized_action is the list of action names instead of numbers 
discretized_action = []
keys = list(d.keys())
for i in range(len(data['action'])):
    for j in range(len(keys)):
        if compare(data['action'][i], keys[j]):
            discretized_action.append(d.get(tuple(data['action'][i])))