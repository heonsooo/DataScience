#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


doc1 = 'new home sales top forecasts'.split()
doc2 = 'home sales rise in july'.split()
doc3 = 'increase in home sales in july'.split()
doc4 ='new home sales rise in November'.split()

doc = []
for i,j,jj,jji in zip(doc1,doc2,doc3,doc4) :
    
    if i not in doc:
        doc.append(i)
    if j not in doc :
        doc.append(j)
    if jj not in  doc :
        doc.append(jj)
    if jji not in  doc :
        doc.append(jji)
    else :
        continue
        
if doc3[-1] not in doc:
    doc.append(doc3[-1])    
if doc4[-1] not in doc:
    doc.append(doc4[-1])

doc = sorted(doc)


# In[2]:


df_i = []
doc_df = doc1+doc2+doc3+doc4

for j in range(0,len(doc)):
    df_i.append(doc_df.count(doc[j]))
    
import math 
idf_i = []
n = len(doc1)+len(doc2)+len(doc3)+len(doc4)
for k in range(0,len(df_i)):
    result = round(math.log10(n/df_i[k]),5)
    idf_i.append(result)


# In[3]:


dataframe = pd.DataFrame({"df_i":df_i, 
                 "idf_i":idf_i}) 
dataframe.index= doc

dataframe

