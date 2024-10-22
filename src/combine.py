#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


sub1 = pd.read_csv('submission1.csv')


# In[3]:


sub2 = pd.read_csv('submission2.csv')


# In[4]:


np.corrcoef([sub1.target,sub2.target])


# In[5]:


sub = sub1.copy()


# In[10]:


sub['target'] = sub1['target'].rank(pct=True)*0.8+sub2['target'].rank(pct=True)*0.2


# In[11]:


sub.to_csv('submission.csv',index=False)


# In[ ]:




