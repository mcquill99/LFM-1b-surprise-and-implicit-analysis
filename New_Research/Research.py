#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats
from scipy import sparse
import scipy
from collections import defaultdict
import implicit
from implicit.als import AlternatingLeastSquares
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVDpp
import rectorch
from rectorch.models.baseline import SLIM, Random, Popularity
from rectorch.models.mf import EASE
from rectorch.data import DataProcessing
from rectorch.samplers import ArrayDummySampler, SparseDummySampler
from rectorch.evaluation import evaluate
from rectorch.utils import collect_results, prepare_for_prediction


# # Data Parsing

# In[11]:


# initialize data
item_threshold = 1 # used to filter out user/artist pairs that have been 
                   #listened to less than the threshold number of times
popular_artist_fraction = 0.2 # top cutoff for what we consider popular artists, in this case the top 20%

user_events_file = 'data/user_events.txt'
low_user_file = 'data/low_main_users.txt'
medium_user_file = 'data/medium_main_users.txt'
high_user_file = 'data/high_main_users.txt'


# In[12]:


#read in user events file
cols = ['user', 'artist', 'album', 'track', 'timestamp']
df_events = pd.read_csv(user_events_file, sep='\t', names=cols)
print('No. of user events: ' + str(len(df_events)))
df_events.head() # check it is all read in properly


# ## User-Artist Matrix

# In[13]:


# create unique user-artist matrix
df_events = df_events.groupby(['user', 'artist']).size().reset_index(name='listens')
print('No. user-artist pairs: ' + str(len(df_events)))
# each row contains a unique user-artist pair, along with how many times the
# user has listened to the artist
#df_events.sort_values(by=['user'])


# In[14]:


df_events.head(1400)


# In[15]:


# filters out artist/user pairs who havent been listened two more than
# item_threshold amount of times to reduce potential load
# kept to 1 currently, so we dont filter out any data
df_events = df_events[df_events['listens'] >= item_threshold] 

# With 1, we see no difference between user-artist pairs here
print('No. filtered user-artist pairs: ' + str(len(df_events))) 

# here, we see the number of unique artists in our matrix
print('No. unique artists: ' + str(len(df_events['artist'].unique())))


# #### How many artists have users listened to?

# In[16]:


# get matrix where each row is a user-id and how many artists they've 
#listened to
user_dist = df_events['user'].value_counts() 

# counts how many unique users there are. prints out user id & a count of how 
# many rows they're included in, which effectively shows how many artists 
# they listen to
num_users = len(user_dist)
print('Mean artists of all users: ' + str(user_dist.mean()))
print('Min artists of all users: ' + str(user_dist.min()))
print('Max artists of all users: ' + str(user_dist.max()))

user_dist.head()


# #### How many users listen to an artist?

# In[17]:


# get artist distribution
# same as previous but with artists, shows artist-id and how many times they
# were listened to buy unique users
artist_dist = df_events['artist'].value_counts()
num_artists = len(artist_dist)
print('No. artists: ' + str(num_artists))
df_events['artist'].value_counts().head


# In[18]:


# get number of  popular artists
num_top_artists = int(popular_artist_fraction * num_artists)

# getting the top top_fraction (0.2) percent of artists, so finding how many
# artists make up 20% of total artists, and then only using the artists those
#number of the most popular aritsts
top_artist_dist = artist_dist[:num_top_artists]
print('No. top artists: ' + str(len(top_artist_dist)))


# In[19]:


# read in users
# user file is just user_id and their mainstreaminess value 
low_users = pd.read_csv(low_user_file, sep=',').set_index('user_id')
medium_users = pd.read_csv(medium_user_file, sep=',').set_index('user_id')
high_users = pd.read_csv(high_user_file, sep=',').set_index('user_id')
num_users = len(low_users) + len(medium_users) + len(high_users)
print('Num users: ' + str(num_users))


# ## Getting Users From Each Popularity Group & Their 10 Most Listened To Artists 

# ### (For Analysis of Streaming Service Algorithmic Bias)

# In[20]:


toList = df_events.loc[df_events['user'] == 42845367].sort_values(by=['listens'], ascending=False)
toList.head() #grabbing random users top 10 artists in 1 of the 3 groups


# In[21]:


to_list_2 = toList['artist'].tolist()[:20]
print(to_list_2)


# # Calculating GAP of User Profiles

# In[22]:


# placeholder vars for numerator of GAPp, waiting to be divided by sie of group
low_gap_p = 0
medium_gap_p = 0
high_gap_p = 0
total_gap_p = 0
#Count for sanity check
low_count = 0
med_count = 0
high_count = 0

for u, df in df_events.groupby('user'):
    
    no_user_artists = len(set(df['artist'])) # profile size //number of artists in users profile
    # get popularity (= fraction of users interacted with item) of user items and calculate average of it
    user_pop_artist_fraq = sum(artist_dist[df['artist']] / num_users) / no_user_artists 
    
    if u in low_users.index: # get user group-specific values
        low_gap_p += user_pop_artist_fraq
        low_count += 1
    elif u in medium_users.index:
        medium_gap_p += user_pop_artist_fraq
        med_count += 1
    else:
        high_gap_p += user_pop_artist_fraq
        high_count += 1

total_gap_p = (low_gap_p + medium_gap_p + high_gap_p) / num_users
low_gap_p /= len(low_users) # average popularity of items/artists in low/med/high groups (gap = group average popularity)
medium_gap_p /= len(medium_users)
high_gap_p /= len(high_users)
print('Low count (for check): ' + str(low_count))
print('Med count (for check): ' + str(med_count))
print('High count (for check): ' + str(high_count))
print(low_gap_p)
print(medium_gap_p)
print(high_gap_p)


# ### Min-Max Scaling Ratings (Not Using Right Now)

# In[ ]:


### Scale listening counts on a scale from 1-1000
"""scaled_df_events = pd.DataFrame()
for user_id, group in df_events.groupby('user'):
    #print(group)
    min_listens = group['listens'].min()
    max_listens = group['listens'].max()
    std = (group['listens'] - min_listens) / (max_listens - min_listens)
    scaled_listens = std * 999 + 1
    to_replace = group.copy()
    to_replace['listens'] = scaled_listens
    #print(to_replace)
    scaled_df_events = scaled_df_events.append(to_replace)
scaled_df_events.head()  """ 
#df_events.groupby('user').head()


# # Rectorch Training

# ### Setting Up The Data



# In[24]:


cfg_data_test = {
    "processing": {
        "data_path": "data_events.csv",
        "threshold": 0,
        "separator": ",",
        "header": None,
        "u_min": 50,
        "i_min": 50
    },
    "splitting": {
        "split_type": "horizontal",
        "sort_by": None,
        "seed": 98765,
        "shuffle": True,
        "valid_size": 0.1,
        "test_size": 0.1,
        "test_prop": 0.2
    }
}



cfg_data_full = {
    "processing": {
        "data_path": "data_events.csv",
        "threshold": 0,
        "separator": ",",
        "header": None,
        "u_min": 0,
        "i_min": 0
    },
    "splitting": {
        "split_type": "horizontal",
        "sort_by": None,
        "seed": 98765,
        "shuffle": True,
        "valid_size": 0.1,
        "test_size": 0.1,
        "test_prop": 0.2
    }
}


# In[25]:


dataset = DataProcessing(cfg_data_test).process_and_split()
dataset


# In[26]:



datasetFull = DataProcessing(cfg_data_full).process_and_split()
datasetFull


# In[18]:


sparse_sampler = SparseDummySampler(dataset, mode="train")




# In[27]:


sparse_sampler_full = SparseDummySampler(datasetFull, mode="train")



# In[28]:


slimFull = SLIM(l1_reg=.0003, l2_reg=.03)
slimFull = slimFull.load_model("slim_full")


# In[ ]:


sparse_sampler_full.test()
results = evaluate(slimFull, sparse_sampler_full, ["ap@5000"])
collect_results(results)


# In[ ]:


results = evaluate(slimFull, sparse_sampler_full, ["auc"])
collect_results(results)





