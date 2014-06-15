
# coding: utf-8

# In[31]:

fh = open('filenames.tsv')


# In[32]:

lines = fh.readlines()


# In[33]:

import os
from glob import glob
target_dir = './testing_ground/'


# In[46]:

def last_increment():
    try:
        return int(sorted([os.path.basename(fname) for fname in glob(target_dir + '[0-9][0-9][0-9]_m2541')])[-1][0:3])
    except (ValueError, IndexError):
        return 0


# In[47]:

dir_list = sorted([os.path.basename(fname) for fname in glob(target_dir+'2541_*')])
for dname in dir_list:
    old_name = os.path.join(target_dir, dname)
    new_name = os.path.join(target_dir, '{0:03d}_m2541'.format(last_increment() + 1))
    os.rename(old_name, new_name)


# In[48]:

dir_list


# In[49]:

last_increment()


# In[ ]:



