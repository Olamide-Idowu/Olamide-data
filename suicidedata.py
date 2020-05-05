#!/usr/bin/env python
# coding: utf-8

# ## WORLD'S MENTAL HEALTH STATE POST COVID19: SUICIDE PREVENTION
# 
# ### SUICIDE DATA PROJECT

# The novel coronavirus (COVID-19) has proven to be a threat to public health and the world's economy. However, one vital area that seems to be ignored is mental health. The importance of mental health for our physical wellbeing and for economy recovery can not be over emphasized. 
# 
# As a Biostatistician and Public Health Data scientist, I am concerned about the mental health state of the world post COVID-19. It is important that proactive measures be taken to avoid surge in rates of mental health illnesses such as depression and suicide. 
# 
# The aim of this project is to use past suicide data to identify the pattern of suicide rates in the world in reference to countries' GDP(representing the health of the economy), sex, age group and generation. 
# 
# It is expected that GDP per capita of most countries would be lower this year (2020) because of the pandemic; in the presence of a relationship between GDP or any other attribute and suicide rates, this project would help develop strategies to ensure suicide dose not become a bigger public health challenge during the pandemic and when it is over. 
# 

# ### Data
# 
# **Data:** The data to be used for this project comprises of information of suicide numbers and rates across different countries of the world from 1985 to 2016. Attributes include country, year, sex, age group, generation, GDP per year, and GDP per capita.
# 
# **Data Source:** The data was downloaded from Kaggle: https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016

# **As a requirement of the challenge, this notebook would contain exploratory data analysis to support my project choice.**
# 
# Please note: Plots will be drawn in this notebook to answer the first two questions (asset 1 and 2) hence the link to this notebook will be uploaded twice.

# **First, let's import the libraries need for the analysis.**

# In[1]:


import pandas as pd
import numpy as np
get_ipython().system(' pip install seaborn')
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

print('Libraries imported')


# Now let's read the data into a pandas dataframe

# In[18]:


df = pd.read_csv(r'C:\Users\USER\Downloads\Suicide\master.csv')
df.head()


# Let's check the shape of the data

# In[3]:


df.shape


# The data has 27,820 rows and 12 columns

# Let's find out if there's any missing data

# In[12]:


missing=df.isnull()
missing.head()


# In[13]:


for column in missing.columns.values.tolist():
    print(column)
    print (missing[column].value_counts())
    print("")


# No missing data, next step is to ensure the attributes contain compatible data types for the analysis, we check information on each type using the dtypes() function.

# In[4]:


df.dtypes


# Checking this, I reaslised the attribute "gdf_for_year ($)" has a space before it and that can be a challenge, it is best to change it. The column: 'HDI for year' would be dropped as it is not useful for this analysis.
# 
# The 'sex' and 'age' attributes are actegorical variables and would be converted to type integer by coding them.

# In[19]:


df.drop(['HDI for year'], axis=1, inplace=True)
df.rename(columns={' gdp_for_year ($) ':'gdp_year ($)'}, inplace=True)
df.columns


# In[26]:


df['age'].unique()


# In[27]:


df.age[df.age == '5-14 years'] = 1
df.age[df.age == '15-24 years'] = 2
df.age[df.age == '25-34 years'] = 3
df.age[df.age == '35-54 years'] = 4
df.age[df.age == '55-74 years'] = 5
df.age[df.age == '75+ years'] = 6

df.head()


# In[28]:


df['sex'].unique()


# In[29]:


df.sex[df.sex == 'male'] = 1
df.sex[df.sex == 'female'] = 0
df.head()


# In[30]:


df.dtypes


# In[31]:


df['sex']=df['sex'].astype(int)
df['age']=df['age'].astype(int)
df.dtypes


# In[32]:


df.describe()


# In[33]:


df.describe(include=['object'])


# In[37]:


df['country'].unique()


# In[39]:


df['country'].nunique()


# The dataset contains data for 101 countries.

# Let's make a copy of the dat which we will manipulate for data visualisation and analysis to avoid making changes to the original data set.

# In[34]:


df1=df.copy()
df1.head()


# Using line plot, let's visualize the average suicide rates (per 100,000 population) from 1985 to 2016

# In[40]:


df1_yr=df1[['year', 'suicides/100k pop']]
df1_yr.head()


# Let's group the data according to the year by the average suicide rate per year for all countries in the data frame. We will also set year as index.

# In[41]:


df1_yr=df1_yr.groupby('year').mean().reset_index()
df1_yr.set_index('year', inplace=True)
df1_yr.head()


# In[43]:


df1_yr.plot(kind='line', figsize=(12, 8))

plt.xlabel('Year') # add to x-label to the plot
plt.ylabel('AVG Suicide Rate (per 100k Population)') # add y-label to the plot
plt.title('Suicide rates from 1985 to 2016') # add title to the plot

plt.show()


# We can see a peak in the rate circa 1995 after which the rate begab to drop with some flunctuations till it reached its lowest point between 2011 and 2012. It increased again and flunctied and then shot up again between 2015 and 2016. The reason for these flunctuations will be examined in this project. 

# Let's plot the suicide rate with GDP per capita to check similarities or differences in the distributions.

# In[44]:


df1_gdp=df1[['year', 'gdp_per_capita ($)']]

df1_gdp.head()


# In[45]:


df1_gdp=df1_gdp.groupby('year').mean().reset_index()
df1_gdp.head()


# In[46]:


df1_gdp.set_index('year', inplace=True)
df1_gdp.head()


# In[47]:


df1_yrmerged = df1_yr.join(df1_gdp, on='year')
df1_yrmerged.head()


# Let's normalize the data so as to give better representation of interation between the two distributions

# In[48]:


df1_yrmerged['suicides/100k pop'] = df1_yrmerged['suicides/100k pop']/df1_yrmerged['suicides/100k pop'].max()
df1_yrmerged['gdp_per_capita ($)'] = df1_yrmerged['gdp_per_capita ($)']/df1_yrmerged['gdp_per_capita ($)'].max()
df1_yrmerged.head()


# In[49]:


df1_yrmerged.plot(kind='line', figsize=(12, 8))
plt.title('Suicide rate and GDP from 1985 to 2016')
plt.ylabel('Suicide rate and GDP')
plt.xlabel('Year')

plt.show()


# Let's use a scatter plot to check the presence of a relationship between suicide rate and GDP

# In[62]:


x = df1_yrmerged['gdp_per_capita ($)']
y = df1_yrmerged['suicides/100k pop']

fit=np.polyfit(x, y, deg=1)

fit


# In[64]:


df1_yrmerged.plot(kind = 'scatter', x = 'gdp_per_capita ($)', y = 'suicides/100k pop', figsize=(12, 8), color='darkblue')

plt.title('Relationship between GDP per capita and AVG suicide rate')
plt.xlabel('GDP per Capita')
plt.ylabel('AVG Suicide Rate')

#plot line of best fit
plt.plot(x, fit[0] * x + fit[1], color = 'red')
plt.annotate('y={0:.0f}x + {1:.0f}'.format(fit[0], fit[1]), xy=(0.70, 0.85))

plt.show()


# Line of best fit shows average suicide rate reducing as GDP increases. 

# In[ ]:


Let's check the distribution of total number of suicides by generation


# In[66]:


df1gen=df1[['suicides_no', 'generation']]
df1gen.head()


# In[67]:


df1gengr=df1gen.groupby('generation').sum().reset_index()
df1gengr.head()


# In[68]:


df1gengr.set_index('generation', inplace=True)
df1gengr.head()


# In[71]:


df1gengr.plot(kind='barh', figsize=(12, 8))
plt.title('suicide nos by generation')
plt.ylabel('suicide nos')
plt.xlabel('Generation')

plt.show()


# Let's check the distribution of number of suicides by sex

# In[72]:


df1sex=df1[['suicides_no', 'sex']]
df1sex.head()


# In[73]:


df1sexgr=df1sex.groupby('sex').sum().reset_index()
df1sexgr.head()


# In[75]:


df1sexgr.set_index('sex', inplace=True)
df1sexgr.head()


# In[77]:


df1sexgr.plot(kind= 'bar', figsize=(10,8))
plt.title('Number of Suicides by Sex')
plt.xlabel('sex (Male=1, Female=0)')
plt.ylabel('Suicides')

plt.show()


# The bar chart above shows that the Male sex committed more suicides than the female sex. To remove possible bias based on population, let's create a bar plot based on suicide rate per 100k population.

# In[78]:


df1sex2 = df1[['sex', 'suicides/100k pop']]
df1sex2.head()


# In[79]:


df1sex2=df1sex2.groupby('sex').mean().reset_index()
df1sex2.head()


# In[80]:


df1sex2.set_index('sex', inplace=True)
df1sex2.head()


# In[81]:


df1sex2.plot(kind = 'bar', figsize=(10, 8))
plt.title('Suicide rate per 100k Population across sexes')
plt.xlabel('Sex (Female =0, Male = 1)')
plt.ylabel('AVG Suicide Rate per 100k Population')

plt.show()


# The bar plot still shows a higher rate of suicide among the male sex than the female sex.

# Let's check the distribution across age groups.

# In[98]:


df1age=df1[['age', 'suicides_no']]
df1age.head()


# In[99]:


df1age=df1age.groupby('age').sum().reset_index()
df1age.head()


# In[85]:


df1age.set_index('age', inplace = True)
df1age.head()


# In[88]:


df1age.plot(kind='bar', figsize=(10, 8))
plt.title('Number of Suicides across Age groups')
plt.xlabel('Age Group (1 = 5-14 yrs, 2= 15-24 yrs, 3= 25-34 yrs, 4= 35-54 yrs, 5= 55-74 yrs, 6= 75+ years)')
plt.ylabel('Suicides')

plt.show()


# The age group (35 - 54 years) has the highest number of suicides from 1985 to 2016.
