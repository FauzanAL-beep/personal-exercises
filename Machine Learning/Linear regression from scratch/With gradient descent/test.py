import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import random

lr = 0.01
def mse(m,b,cso):
    res = 0
    for i in cso:
        res = res +((i[1]-((m*i[0])+b))*(i[1]-((m*i[0])+b)))
    return res/len(cso)
def jm(m,b,cso,lr):
    res = 0
    for i in cso:
        res = res +((i[1]-((m*i[0])+b))*i[0]*(-2))
    return (res/len(cso))*lr
def jb(m,b,cso,lr):
    res = 0
    for i in cso:
        res = res +((i[1]-((m*i[0])+b))*(-2))
    return (res/len(cso))*lr
results = []
with open("./datasets/idk.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)
slope = random.uniform(0, 1)
b = random.uniform(0, 1)

fail = 0

'''
fail = mse(slope,b,results)
print(fail)
slopet = slope
bt = b
slope = slopet - jm(slopet,bt,results,lr)
b = bt - jb(slopet,bt,results,lr)
print(slope)
print(b)
print('---')

fail = mse(slope,b,results)
print(fail)
slopet = slope
bt = b
slope = slopet - jm(slopet,bt,results,lr)
b = bt - jb(slopet,bt,results,lr)
print(slope)
print(b)
print('---')

fail = mse(slope,b,results)
print(fail)
slopet = slope
bt = b
slope = slopet - jm(slopet,bt,results,lr)
b = bt - jb(slopet,bt,results,lr)
print(slope)
print(b)
print('---')
'''

while True:
	fail = mse(slope,b,results)
	print(fail)
	if fail<0.5:
		break
	else:
		slopet = slope
		bt = b
		slope = slopet - jm(slopet,bt,results,lr)
		b = bt - jb(slopet,bt,results,lr)

print(fail)


'''
while True:
    fail = mse(slope,b,results)
    print(fail)
    if fail<0.5:
        break
    else:
        j = fail*2
        slope = slope - j
        b = b - j

print(fail)
'''