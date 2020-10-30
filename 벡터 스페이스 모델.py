#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd

doc1 = 'new home sales top forecasts'.split()
doc2 = 'home sales rise in july'.split()
doc3 = 'increase in home sales in july'.split()
doc4 ='new home sales rise in November'.split()

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


# 이걸 어떻게 계산할까.. 
doc_1 = [0,1,1,0,0,0,1,0,1,1]

doc_2 = [0,0,1,1,0,1,0,1,1,0]

doc_3 = [0,0,1,2,1,1,0,0,1,0]

doc_4 = [1,0,1,1,0,0,1,1,1,0]


# In[ ]:


df=pd.DataFrame({"doc1":doc_1, 
                 "doc2":doc_2,
                 "doc3":doc_3,
                 "doc4":doc_4}) 
df.index= doc


doc_1_sqr ,doc_2_sqr ,doc_3_sqr ,doc_4_sqr = 0,0,0,0

# 각 리스트의 크기(벡터의 크기, 다시 말해서 절댓값^2 )  1부터 4까지 
for i,j,k,l in zip(doc_1,doc_2,doc_3, doc_4) :
    doc_1_sqr += i**2 
    doc_2_sqr += j**2 
    doc_3_sqr += k**2 
    doc_4_sqr += l**2 
    

    
    

sim_d1_d2 = np.dot(doc_1,doc_2)/(doc_1_sqr*doc_2_sqr) 
sim_d1_d3 = np.dot(doc_1,doc_3)/(doc_1_sqr*doc_3_sqr) 
sim_d1_d4 = np.dot(doc_1,doc_4)/(doc_1_sqr*doc_4_sqr) 

sim_d2_d3 = np.dot(doc_2,doc_3)/(doc_2_sqr*doc_3_sqr) 
sim_d2_d4 = np.dot(doc_2,doc_4)/(doc_2_sqr*doc_4_sqr) 

sim_d3_d4 = np.dot(doc_3,doc_4)/(doc_3_sqr*doc_4_sqr) 



similiarty = [sim_d1_d2 , sim_d1_d3 , sim_d1_d4, sim_d2_d3, sim_d2_d4, sim_d3_d4]

print(similiarty) 

print(sorted(similiarty))


# # for문 이용해서 코드 짧게

# In[ ]:



doc_n = [doc_1,doc_2,doc_3, doc_4 ]
doc_n_sqr = [doc_1_sqr ,doc_2_sqr ,doc_3_sqr ,doc_4_sqr]

final_result = []

for i in range(0,3):
    for j in range(i+1,4):
        result = round(np.dot(doc_n[i],doc_n[j])/(doc_n_sqr[i]*doc_n_sqr[j]),4)
        name = 'sim_d'+str(i+1)+'_d'+str(j+1)
        tup = (name, result)
        final_result.append(tup)
        
print(final_result, '\n')

print(sorted(final_result, reverse=True)) 
        
        
    
    


# In[ ]:




