#!/usr/bin/env python
# coding: utf-8

# In[9]:


#question-1
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/astroclubiitk/computational-astrophysics/main/Week-1/Assignment/Starfish_Data.csv')
df['d'] = 10**((df['#Vmag'] + 2.5*df['logL'] + 0.17)/5)
avg_dist = (df['d'] * df['Prob']).sum()/df['Prob'].sum()
print(avg_dist)


# In[10]:


#question-2(1)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/astroclubiitk/computational-astrophysics/main/Week-1/Assignment/Astrosat_Catalog.csv')
def plot_mwd(RA,Dec):
    x = np.remainder(RA+360,360)
    ind = x>180
    x[ind] -=360    
    x=-x
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360,360)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection = 'mollweide', facecolor = 'k')
    ax.scatter(np.radians(x),np.radians(Dec), marker = '*', color = 'r')
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("RA")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("DEC")
    ax.yaxis.label.set_fontsize(12)
hmxb = df.where(df['Final_Type'] == 'HMXB')
lmxb = df.where(df['Final_Type'] == 'LMXB')
ra1 = hmxb['ra']
dec1 = hmxb['dec']
ra2 = lmxb['ra']
dec2 = lmxb['dec']
plot_mwd(ra1,dec1)
plt.title('HMXB')
plt.grid(True)
plot_mwd(ra2,dec2)
plt.title('LMXB')
plt.grid(True)


# In[11]:


#question-2(2)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/astroclubiitk/computational-astrophysics/main/Week-1/Assignment/Astrosat_Catalog.csv')
def plot_mwd(RA,DEC):
    x = np.remainder(RA+360,360)
    ind = x>180
    x[ind] -=360    
    x=-x
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360,360)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection = 'mollweide', facecolor = 'k')
    ax.scatter(np.radians(x),np.radians(DEC), marker = '*', color = 'r')
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("RA")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("DEC")
    ax.yaxis.label.set_fontsize(12)
flag1 = df.where(df['Astrosat_Flag'] == 1)
flag0 = df.where(df['Astrosat_Flag'] == 0)
ra0 = flag0['ra']
dec0 = flag0['dec']
ra1 = flag1['ra']
dec1 = flag1['dec']
plot_mwd(ra0,dec0)
plt.title('Not Observed by Astrosat')
plt.grid(True)
plot_mwd(ra1,dec1)
plt.title('Observed by Astrosat')
plt.grid(True)


# In[12]:


#question-2(3a)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/astroclubiitk/computational-astrophysics/main/Week-1/Assignment/Astrosat_Catalog.csv')
def plot_mwd(RA1,Dec1,RA2,Dec2):
    x1 = np.remainder(RA1+360,360)
    ind = x1>180
    x1[ind] -=360    
    x1=-x1
    x2 = np.remainder(RA2+360,360)
    ind = x2>180
    x2[ind] -=360    
    x2=-x2
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360,360)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection = 'mollweide', facecolor = 'k')
    ax.scatter(np.radians(x1),np.radians(Dec1), marker = '*', color = 'r', label = 'HMXB')
    ax.scatter(np.radians(x2),np.radians(Dec2), marker = '*', color = 'w', label = 'LMXB')
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("RA")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("DEC")
    ax.yaxis.label.set_fontsize(12)
hmxb = df.where(df['Final_Type'] == 'HMXB')
lmxb = df.where(df['Final_Type'] == 'LMXB')
ra1 = hmxb['ra']
dec1 = hmxb['dec']
ra2 = lmxb['ra']
dec2 = lmxb['dec']
plot_mwd(ra1,dec1,ra2,dec2)
plt.title('HMXB and LMXB')
plt.grid(True)
plt.legend()


# In[13]:


#question-2(3b)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/astroclubiitk/computational-astrophysics/main/Week-1/Assignment/Astrosat_Catalog.csv')
def plot_mwd(RA0,DEC0,RA1,DEC1):
    x0 = np.remainder(RA0+360,360)
    ind = x0>180
    x0[ind] -=360    
    x0=-x0
    x1 = np.remainder(RA1+360,360)
    ind = x1>180
    x1[ind] -=360    
    x1=-x1
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360,360)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection = 'mollweide', facecolor = 'k')
    ax.scatter(np.radians(x0),np.radians(DEC0), marker = '*', color = 'r', label = 'not observed')
    ax.scatter(np.radians(x1),np.radians(DEC1), marker = '*', color = 'w', label = 'observed')
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("RA")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("DEC")
    ax.yaxis.label.set_fontsize(12)
flag1 = df.where(df['Astrosat_Flag'] == 1)
flag0 = df.where(df['Astrosat_Flag'] == 0)
ra0 = flag0['ra']
dec0 = flag0['dec']
ra1 = flag1['ra']
dec1 = flag1['dec']
plot_mwd(ra0,dec0,ra1,dec1)
plt.title('Observed and Not Observed by Astrosat')
plt.grid(True)
plt.legend()


# In[ ]:




