#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd

doc1 = ['new', 'home','sales','top','forecasts']
doc2 = ['home','sales','rise','in','july']
doc3 = ['increase', 'in','home','sales','in' ,'july']
doc4 =['new', 'home','sales','rise','in','November']

print('\n',sorted(doc1),'\n', sorted(doc2),'\n',sorted(doc3),'\n',sorted(doc4),'\n')

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
print(doc,'\n' ,len(doc))

doc_1 = [0,1,1,0,0,0,1,0,1,1]

doc_2 = [0,0,1,1,0,1,0,1,1,0]

doc_3 = [0,0,1,2,1,1,0,0,1,0]

doc_4 = [1,0,1,1,0,0,1,1,1,0]








# In[30]:


df=pd.DataFrame({"doc1":doc_1, 
                 "doc2":doc_2,
                 "doc3":doc_3,
                 "doc4":doc_4}) 
df.index= doc

p = np.dot(doc_1,doc_2)

p

# 각 리스트의 크기(벡터 크기)  1부터 4까지 
for i in doc_1 :
    n += i**2 
    
    print(n)


# In[ ]:




