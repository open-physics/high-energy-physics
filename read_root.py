''' ############################################################### '''
''' Read root file and convert into Data frame and then to csv file '''
''' ############################################################### '''
import time
import uproot
from pprint import pprint
import pandas as pd
import polars as pl
import numpy as np

from config import input_rootfile

file_root = uproot.open(input_rootfile)
# pprint(file_root.keys())
# pprint(file_root.values())

''' option 1: to convert the root file to dataframe and finaly to csv '''
dfs = []
tree = file_root["fDBEvtTree"] # name of Tree in root file

t0 = time.time() # start time
for key in tree.keys():
    # print(key, tree[key].array())
    # print(key, len(tree[key].array()))

    value = tree[key].array()
    df = pd.DataFrame({key:value})
    dfs.append(df)

dataframe = pd.concat(dfs, axis=1)  #ignore_index=False

t1 = time.time() # stop time
print(t1-t0)

print("data frame:", dataframe)



''' option 2: to convert the root file to dataframe '''
t2 = time.time() # start time

sdf = []
for key in tree.keys():
    value = tree[key].array()
    # df = pd.DataFrame({key:value})
    # dfs.append(df)
    sdf.append(value)
sdf = pd.DataFrame(np.array(sdf).T, columns=tree.keys())
t3 = time.time()
print(t3-t2)
print("dataframe:", sdf)


# try polars

t4 = time.time() # start time
''' option 3: to convert the root file to dataframe '''

ldf = []
for key in tree.keys():
    value = tree[key].array()
    ldf.append(value)
ldf = pl.DataFrame(np.array(ldf).T, columns=tree.keys())

t5 = time.time() # stop time
print(t5-t4)
print("data frame:", ldf)


''' Save the data frame to csv file format'''
# dataframe.to_csv("amptsm.csv")


