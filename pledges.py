#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('default')
from sklearn.linear_model import LinearRegression
import datetime as dt
from datetime import datetime, timedelta
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import ttest_1samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import ttest_ind
from scipy.stats import chi2
import tikzplotlib 
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
df = pd.read_csv('pledges.csv')


# In[3]:


df['DATE'] = pd.to_datetime(df['DATE'],format="%d/%m/%Y")
df['ETHNICITY']=df['ETHNICITY'].replace('Malay', 'Bumiputera')
df['ETHNICITY']=df['ETHNICITY'].replace('Orang Asli', 'Bumiputera')
df['AGE']=df['AGE'].replace('Above 99', 'Above 80')
df['AGE']=df['AGE'].replace('80-89', 'Above 80')
df['AGE']=df['AGE'].replace('90-99', 'Above 80')
df['AGE']=df['AGE'].replace('13-17', '00-19')
df['AGE']=df['AGE'].replace('00-04', '00-19')
df['AGE']=df['AGE'].replace('05-12', '00-19')
df['AGE']=df['AGE'].replace('18-19', '00-19')


# In[480]:


df


# In[4]:


df = df[df != '-1']
df.dropna(inplace=True)
df.fillna(0, inplace=True)
df


# In[484]:


# Group the data by date and sum the daily counts
daily_counts = df.groupby('DATE')['COUNT'].sum()

# Create a line plot
plt.plot(daily_counts.index, daily_counts)
plt.xlabel('Date')
plt.ylabel('Total Daily Count')
tikzplotlib.save("tikz1.tex")


# In[5]:


# Group the data by STATE and DATE and calculate the sum of COUNT for each group

df_state = df.groupby(['STATE', 'DATE'])['COUNT'].sum().reset_index().pivot(index='DATE', columns='STATE', values='COUNT')
df_state.fillna(0, inplace=True)

# Plot the data as multiple lines
ax = df_state.plot(kind='line', figsize=(12,6), title='Total Count by State')
ax.set_xlabel('Date')
ax.set_ylabel('Total Count')
plt.show()


# In[129]:


# Group the data by STATE and DATE and calculate the sum of COUNT for each group

df_eth = df.groupby(['ETHNICITY', 'DATE'])['COUNT'].sum().reset_index().pivot(index='DATE', columns='ETHNICITY', values='COUNT')
df_eth.fillna(0, inplace=True)

# Plot the data as multiple lines
ax = df_eth.plot(kind='line', figsize=(12,6), title='Total Count by Ethnicity')
ax.set_xlabel('Date')
ax.set_ylabel('Total Count')
plt.show()


# In[136]:


# Group the data by STATE and DATE and calculate the sum of COUNT for each group

df_age = df.groupby(['AGE', 'DATE'])['COUNT'].sum().reset_index().pivot(index='DATE', columns='AGE', values='COUNT')
df_age.fillna(0, inplace=True)

# Plot the data as multiple lines
ax = df_age.plot(kind='line', figsize=(12,6), title='Total Count by Age')
ax.set_xlabel('Date')
ax.set_ylabel('Total Count')
plt.show()


# In[41]:


# Group the data by STATE and DATE and calculate the sum of COUNT for each group

df_male = df.groupby(['MALE', 'DATE'])['COUNT'].sum().reset_index().pivot(index='DATE', columns='MALE', values='COUNT')
df_male.fillna(0, inplace=True)

# Plot the data as multiple lines
ax = df_male.plot(kind='line', figsize=(12,6), title='Total Count by Male')
ax.set_xlabel('Date')
ax.set_ylabel('Total Count')
plt.show()


# In[598]:


def adf_lb(data):
    result = adfuller(data)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'{key}, {value}')
    plot_acf(data, lags=30)
    plt.show()
    plot_pacf(data, lags=30)
    plt.show()
    lb_stat, lb_pvalue = sm.stats.acorr_ljungbox(data, lags=[30], return_df=False)
    print('Ljung-Box Q-statistic:', lb_stat[0])
    print('Ljung-Box p-value:', lb_pvalue[0])
    print('\n')
    
    adf.append(result[0])
    adfp.append(result[1])
    lbq.append(lb_stat[0])
    lbp.append(lb_pvalue[0])


# In[609]:


df_state_train = df_state.loc['2022-03-23':'2022-09-06']
df_sal=pd.DataFrame()
df_sal['state']=df_state.columns
adf=[]
adfp=[]
lbq=[]
lbp=[]

for state in df_state_train.columns:
    print(str(state))
    adf_lb(df_state_train[state])
    
df_sal['ADF-statistics']=adf
df_sal['ADF p-value']=adfp
df_sal['Ljung-Box Q-stat']=lbq
df_sal['Ljung-Box p-value']=lbp
df_sal.set_index('state', inplace=True)
df_sal.to_latex('table1.tex')


# In[610]:


df_eth_train = df_eth.loc['2022-03-23':'2022-09-06']
df_eal=pd.DataFrame()
df_eal['eth']=df_eth.columns
adf=[]
adfp=[]
lbq=[]
lbp=[]

for eth in df_eth_train.columns:
    print(str(eth))
    adf_lb(df_eth_train[eth])
    
df_eal['ADF-statistics']=adf
df_eal['ADF p-value']=adfp
df_eal['Ljung-Box Q-stat']=lbq
df_eal['Ljung-Box p-value']=lbp
df_eal.set_index('eth', inplace=True)
df_eal.to_latex('table2.tex')


# In[608]:


df_age_train = df_age.loc['2022-03-23':'2022-09-06']
df_aal=pd.DataFrame()
df_aal['age']=df_age.columns
adf=[]
adfp=[]
lbq=[]
lbp=[]

for age in df_age_train.columns:
    print(str(age))
    adf_lb(df_age_train[age])
    
df_aal['ADF-statistics']=adf
df_aal['ADF p-value']=adfp
df_aal['Ljung-Box Q-stat']=lbq
df_aal['Ljung-Box p-value']=lbp
df_aal.set_index('age', inplace=True)
df_aal.to_latex('table3.tex')


# In[735]:


def arwn(data):
    # Fit AR(1) model
    ar_1 = sm.tsa.ARIMA(data, order=(1,1,0)).fit()

    # Print model summary
    print(ar_1.summary())

    # Test model fit
    sm.stats.diagnostic.acorr_ljungbox(ar_1.resid, lags=[20], return_df=True)

    result = adfuller(ar_1.resid)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
        
    ar1.append(ar_1.summary().tables[1][2][1])
    aic_1.append(ar_1.aic)
    bic_1.append(ar_1.bic)
    jb_1.append(ar_1.summary().tables[2][1][3])
    het_1.append(ar_1.summary().tables[2][3][1])
    lb_1.append(ar_1.summary().tables[2][1][1])
    adf_1.append(result[1])
    
        
    # Fit white noise model
    wn = sm.tsa.ARIMA(data, order=(2,0,0)).fit()

    # Print model summary
    print(wn.summary())

    # Test model fit
    sm.stats.diagnostic.acorr_ljungbox(wn.resid, lags=[20], return_df=True)

    result = adfuller(wn.resid)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
        
    aic_0.append(wn.aic)
    bic_0.append(wn.bic)
    jb_0.append(wn.summary().tables[2][1][3])
    het_0.append(wn.summary().tables[2][3][1])
    lb_0.append(wn.summary().tables[2][1][1])
    adf_0.append(result[1])
    print('\n')


# In[736]:


df_state_train = df_state.loc['2022-03-23':'2022-09-06']
df_sfit1 = pd.DataFrame()
df_sfit0 = pd.DataFrame()
df_sfit1['state']=df_state.columns
df_sfit0['state']=df_state.columns
ar1=[]
aic_1=[]
bic_1=[]
jb_1=[]
het_1=[]
lb_1=[]
adf_1=[]
aic_0=[]
bic_0=[]
jb_0=[]
het_0=[]
lb_0=[]
adf_0=[]

for state in df_state_train.columns:
    print(str(state))
    arwn(df_state_train[state])
    
df_sfit1['AR_1']=ar1
df_sfit1['AR1_AIC']=aic_1
df_sfit1['AR1_BIC']=bic_1
df_sfit1['AR1_JB']=jb_1
df_sfit1['AR1_HET']=het_1
df_sfit1['AR1_LB']=lb_1
df_sfit1['AR1_ADF']=adf_1
df_sfit0['AR0_AIC']=aic_0
df_sfit0['AR0_BIC']=bic_0
df_sfit0['AR0_JB']=jb_0
df_sfit0['AR0_HET']=het_0
df_sfit0['AR0_LB']=lb_0
df_sfit0['AR0_ADF']=adf_0
df_sfit1.set_index('state',inplace=True)
df_sfit0.set_index('state',inplace=True)
df_sfit1.to_latex('table41.tex')
df_sfit0.to_latex('table40.tex')


# In[732]:


df_eth_train = df_eth.loc['2022-03-23':'2022-09-06']
df_efit1 = pd.DataFrame()
df_efit0 = pd.DataFrame()
df_efit1['eth']=df_eth.columns
df_efit0['eth']=df_eth.columns
ar1=[]
aic_1=[]
bic_1=[]
jb_1=[]
het_1=[]
lb_1=[]
adf_1=[]
aic_0=[]
bic_0=[]
jb_0=[]
het_0=[]
lb_0=[]
adf_0=[]

for eth in df_eth_train.columns:
    print(str(eth))
    arwn(df_eth_train[eth])
    
df_efit1['AR_1']=ar1
df_efit1['AR1_AIC']=aic_1
df_efit1['AR1_BIC']=bic_1
df_efit1['AR1_JB']=jb_1
df_efit1['AR1_HET']=het_1
df_efit1['AR1_LB']=lb_1
df_efit1['AR1_ADF']=adf_1
df_efit0['AR0_AIC']=aic_0
df_efit0['AR0_BIC']=bic_0
df_efit0['AR0_JB']=jb_0
df_efit0['AR0_HET']=het_0
df_efit0['AR0_LB']=lb_0
df_efit0['AR0_ADF']=adf_0
df_efit1.set_index('eth',inplace=True)
df_efit0.set_index('eth',inplace=True)
df_efit1.to_latex('table51.tex')
df_efit0.to_latex('table50.tex')


# In[733]:


df_age_train = df_age.loc['2022-03-23':'2022-09-06']
df_afit1 = pd.DataFrame()
df_afit0 = pd.DataFrame()
df_afit1['age']=df_age.columns
df_afit0['age']=df_age.columns
ar1=[]
aic_1=[]
bic_1=[]
jb_1=[]
het_1=[]
lb_1=[]
adf_1=[]
aic_0=[]
bic_0=[]
jb_0=[]
het_0=[]
lb_0=[]
adf_0=[]

for age in df_age_train.columns:
    print(str(age))
    arwn(df_age_train[age])
    
df_afit1['AR_1']=ar1
df_afit1['AR1_AIC']=aic_1
df_afit1['AR1_BIC']=bic_1
df_afit1['AR1_JB']=jb_1
df_afit1['AR1_HET']=het_1
df_afit1['AR1_LB']=lb_1
df_afit1['AR1_ADF']=adf_1
df_afit0['AR0_AIC']=aic_0
df_afit0['AR0_BIC']=bic_0
df_afit0['AR0_JB']=jb_0
df_afit0['AR0_HET']=het_0
df_afit0['AR0_LB']=lb_0
df_afit0['AR0_ADF']=adf_0
df_afit1.set_index('age',inplace=True)
df_afit0.set_index('age',inplace=True)
df_afit1.to_latex('table60.tex')
df_afit0.to_latex('table61.tex')


# In[61]:


# 00-19(AR), 20-29(cincai), 30-39(AR), 40-49(WN but cincai), 50-59(WN), 60-69(AR), 70-79(WN)# Indian, others is more WN


# In[737]:


df_state_train = df_state.loc['2022-03-23':'2022-09-30']
df_sfit1 = pd.DataFrame()
df_sfit0 = pd.DataFrame()
df_sfit1['state']=df_state.columns
df_sfit0['state']=df_state.columns
ar1=[]
aic_1=[]
bic_1=[]
jb_1=[]
het_1=[]
lb_1=[]
adf_1=[]
aic_0=[]
bic_0=[]
jb_0=[]
het_0=[]
lb_0=[]
adf_0=[]

for state in df_state_train.columns:
    print(str(state))
    arwn(df_state_train[state])

df_sfit1['AR_1']=ar1
df_sfit1['AR1_AIC']=aic_1
df_sfit1['AR1_BIC']=bic_1
df_sfit1['AR1_JB']=jb_1
df_sfit1['AR1_HET']=het_1
df_sfit1['AR1_LB']=lb_1
df_sfit1['AR1_ADF']=adf_1
df_sfit0['AR0_AIC']=aic_0
df_sfit0['AR0_BIC']=bic_0
df_sfit0['AR0_JB']=jb_0
df_sfit0['AR0_HET']=het_0
df_sfit0['AR0_LB']=lb_0
df_sfit0['AR0_ADF']=adf_0
df_sfit1.set_index('state',inplace=True)
df_sfit0.set_index('state',inplace=True)
df_sfit1.to_latex('table71.tex')
df_sfit0.to_latex('table70.tex')

df_age_train = df_age.loc['2022-03-23':'2022-09-30']
df_afit1 = pd.DataFrame()
df_afit0 = pd.DataFrame()
df_afit1['age']=df_age.columns
df_afit0['age']=df_age.columns
ar1=[]
aic_1=[]
bic_1=[]
jb_1=[]
het_1=[]
lb_1=[]
adf_1=[]
aic_0=[]
bic_0=[]
jb_0=[]
het_0=[]
lb_0=[]
adf_0=[]

for age in df_age_train.columns:
    print(str(age))
    arwn(df_age_train[age])
    
df_afit1['AR_1']=ar1    
df_afit1['AR1_AIC']=aic_1
df_afit1['AR1_BIC']=bic_1
df_afit1['AR1_JB']=jb_1
df_afit1['AR1_HET']=het_1
df_afit1['AR1_LB']=lb_1
df_afit1['AR1_ADF']=adf_1
df_afit0['AR0_AIC']=aic_0
df_afit0['AR0_BIC']=bic_0
df_afit0['AR0_JB']=jb_0
df_afit0['AR0_HET']=het_0
df_afit0['AR0_LB']=lb_0
df_afit0['AR0_ADF']=adf_0
df_afit1.set_index('age',inplace=True)
df_afit0.set_index('age',inplace=True)
df_afit1.to_latex('table91.tex')
df_afit0.to_latex('table90.tex')

df_eth_train = df_eth.loc['2022-03-23':'2022-09-30']
df_efit1 = pd.DataFrame()
df_efit0 = pd.DataFrame()
df_efit1['eth']=df_eth.columns
df_efit0['eth']=df_eth.columns
ar1=[]
aic_1=[]
bic_1=[]
jb_1=[]
het_1=[]
lb_1=[]
adf_1=[]
aic_0=[]
bic_0=[]
jb_0=[]
het_0=[]
lb_0=[]
adf_0=[]

for eth in df_eth_train.columns:
    print(str(eth))
    arwn(df_eth_train[eth])
    
df_efit1['AR_1']=ar1    
df_efit1['AR1_AIC']=aic_1
df_efit1['AR1_BIC']=bic_1
df_efit1['AR1_JB']=jb_1
df_efit1['AR1_HET']=het_1
df_efit1['AR1_LB']=lb_1
df_efit1['AR1_ADF']=adf_1
df_efit0['AR0_AIC']=aic_0
df_efit0['AR0_BIC']=bic_0
df_efit0['AR0_JB']=jb_0
df_efit0['AR0_HET']=het_0
df_efit0['AR0_LB']=lb_0
df_efit0['AR0_ADF']=adf_0
df_efit1.set_index('eth',inplace=True)
df_efit0.set_index('eth',inplace=True)
df_efit1.to_latex('table81.tex')
df_efit0.to_latex('table80.tex')


# In[23]:


def graph(data_train, data_test):

    # Fit the AR(1) model on the train set
    model = ARIMA(data_train, order=(1, 0, 0))
    model_fit = model.fit()

    forecast = data_train

    # Iterate through each date in the forecast period
    for date in pd.date_range(start='2022-09-07', end='2023-03-23', freq='D'):
    
        # Predict the next value in the sequence
        pred = model_fit.forecast()
        forecast = forecast.append(pred)
       
        # Retrain the model on the updated train set
        model = ARIMA(forecast, order=(1, 0, 0))
        model_fit = model.fit()

    # Create a dataframe to hold the forecast values
    actual = pd.concat([data_train, data_test], axis=0)

    # Plot the actual and forecast values
    plt.plot(actual, label='Actual')
    plt.plot(forecast, label='Forecast')
    plt.legend()
    plt.show()
    #tikzplotlib.save("tikz2.tex")
    
    # Calculate the abnormal returns
    abnormal_returns = actual - forecast

    # Plot the abnormal returns
    plt.plot(abnormal_returns.loc['2022-09-07':'2022-09-13'])
    plt.title('Abnormal Returns')
    plt.xlabel('Date')
    plt.ylabel('Abnormal Returns')
    #plt.show()
    tikzplotlib.save("tikz3.tex")

    # Perform t-test on abnormal returns
    t_stat, p_value = ttest_1samp(abnormal_returns.loc['2022-09-07':'2022-09-13'], 0)
    print(abnormal_returns.loc['2022-09-07':'2022-09-13'].mean())
    print((str(t_stat)+', '+ str(p_value)))
    print('\n')

    #shock1.append(abnormal_returns.loc['2022-09-07':'2022-09-13'].mean())
    #t1.append(t_stat)
    #p1.append(p_value)


# In[363]:


# Create a dictionary to store the results
df_sar=pd.DataFrame()
df_sar['state']=df_state.columns
shock1=[]
t1=[]
p1=[]

for state in df_state.columns: 
        print(str(state))
        graph(df_state.loc['2022-03-23':'2022-09-06'][state], df_state.loc['2022-09-07':'2023-03-23'][state])

df_sar['shock1']=shock1
df_sar['t1']=t1
df_sar['p1']=p1


# In[139]:


df_ear=pd.DataFrame()
df_ear['eth']=df_eth.columns
shock1=[]
t1=[]
p1=[]

for eth in df_eth.columns: 
        print(str(eth))
        graph(df_eth.loc['2022-03-23':'2022-09-06'][eth], df_eth.loc['2022-09-07':'2023-03-23'][eth])

df_ear['shock1']=shock1
df_ear['t1']=t1
df_ear['p1']=p1


# In[300]:


df_aar=pd.DataFrame()
df_aar['age']=df_age.columns
shock1=[]
t1=[]
p1=[]

for age in df_age.columns: 
        print(str(age))
        graph(df_age.loc['2022-03-23':'2022-09-06'][age], df_age.loc['2022-09-07':'2023-03-23'][age])

df_aar['shock1']=shock1
df_aar['t1']=t1
df_aar['p1']=p1


# In[751]:


graph2(df_age.loc['2022-03-23':'2022-09-30']['40-49'], df_age.loc['2022-10-01':'2023-03-23']['40-49'])
graph2(df_age.loc['2022-03-23':'2022-09-30']['50-59'], df_age.loc['2022-10-01':'2023-03-23']['50-59'])
graph2(df_age.loc['2022-03-23':'2022-09-30']['60-69'], df_age.loc['2022-10-01':'2023-03-23']['60-69'])
df_aar


# In[24]:


def graph2(data_train, data_test):

    # Fit the AR(1) model on the train set
    model = ARIMA(data_train, order=(2, 0, 0))
    model_fit = model.fit()

    forecast = data_train

    # Iterate through each date in the forecast period
    for date in pd.date_range(start='2022-10-01', end='2023-03-23', freq='D'):
    
        # Predict the next value in the sequence
        pred = model_fit.forecast()
        forecast = forecast.append(pred)
       
        # Retrain the model on the updated train set
        model = ARIMA(forecast, order=(2, 0, 0))
        model_fit = model.fit()

    # Create a dataframe to hold the forecast values
    actual = pd.concat([data_train, data_test], axis=0)

    # Plot the actual and forecast values
    plt.plot(actual, label='Actual')
    plt.plot(forecast, label='Forecast')
    plt.legend()
    plt.show()
    #tikzplotlib.save("tikz4.tex")
    
    # Calculate the abnormal returns
    abnormal_returns = actual - forecast

    # Plot the abnormal returns
    plt.plot(abnormal_returns.loc['2022-10-01':'2022-10-07'])
    plt.title('Abnormal Returns')
    plt.xlabel('Date')
    plt.ylabel('Abnormal Returns')
    #plt.show()
    tikzplotlib.save('tikz5.tex')

    # Perform t-test on abnormal returns
    t_stat, p_value = ttest_1samp(abnormal_returns.loc['2022-10-01':'2022-10-07'], 0)
    print(abnormal_returns.loc['2022-10-01':'2022-10-07'].mean())
    print((str(t_stat)+', '+ str(p_value)))
    print('\n')
    
    #shock2.append(abnormal_returns.loc['2022-10-01':'2022-10-07'].mean())
    #t2.append(t_stat)
    #p2.append(p_value)


# In[364]:


shock2=[]
t2=[]
p2=[]

for state in df_state.columns: 
        print(str(state))
        graph2(df_state.loc['2022-03-23':'2022-09-30'][state], df_state.loc['2022-10-01':'2023-03-23'][state])

df_sar['shock2']=shock2
df_sar['t2']=t2
df_sar['p2']=p2


# In[659]:


pop = [4009670,2071900,2001000,932700,1098500,1675000,2500000,254400,1783000,3900000,2907500,6518500,1280000,1982112,99500,91900]
df_sar['pop']=[x/10000 for x in pop]
df_sar['%shock1']=[df_sar['shock1'][i] / df_sar['pop'][i] for i in range(len(pop))]
df_sar['%shock2']=[df_sar['shock2'][i] / df_sar['pop'][i] for i in range(len(pop))]
df_sar.drop(columns='pop').to_latex('table7.tex')


# In[144]:


shock2=[]
t2=[]
p2=[]

for eth in df_eth.columns: 
        print(str(eth))
        graph2(df_eth.loc['2022-03-23':'2022-09-30'][eth], df_eth.loc['2022-10-01':'2023-03-23'][eth])

df_ear['shock2']=shock2
df_ear['t2']=t2
df_ear['p2']=p2


# In[658]:


epop=[23248797,7771932,2244480,2344979]
df_ear['pop']=[x/10000 for x in epop]
df_ear['%shock1']=[df_ear['shock1'][i] / df_ear['pop'][i] for i in range(len(epop))]
df_ear['%shock2']=[df_ear['shock2'][i] / df_ear['pop'][i] for i in range(len(epop))]
df_ear.drop(columns='pop').to_latex('table8.tex')


# In[302]:


shock2=[]
t2=[]
p2=[]

for age in df_age.columns: 
        print(str(age))
        graph2(df_age.loc['2022-03-23':'2022-09-30'][age], df_age.loc['2022-10-01':'2023-03-23'][age])

df_aar['shock2']=shock2
df_aar['t2']=t2
df_aar['p2']=p2


# In[687]:


df_aar.drop(columns='pop').to_latex('table9.tex')
df_aar


# In[677]:


for cat in df_pop.columns:
    print(str(cat))
    print(df_pop[cat].nunique())


# In[675]:


df_pop = pd.read_parquet('population_2022.parquet')
df_pop = df_pop[df_pop['sex'] != 'overall']
df_pop = df_pop[df_pop['ethnicity'] != 'overall']
df_pop['age']=df_pop['age'].replace(['0-4','5-9','10-14','15-19'],'00-19')
df_pop['age']=df_pop['age'].replace(['20-24','25-29'],'20-29')
df_pop['age']=df_pop['age'].replace(['30-34','35-39'],'30-39')
df_pop['age']=df_pop['age'].replace(['40-44','45-49'],'40-49')
df_pop['age']=df_pop['age'].replace(['50-54','55-59'],'50-59')
df_pop['age']=df_pop['age'].replace(['60-64','65-69'],'60-69')
df_pop['age']=df_pop['age'].replace(['70-74','75-79'],'70-79')
df_pop['age']=df_pop['age'].replace(['80-84','85+'],'Above 80')
df_pop['ethnicity']=df_pop['ethnicity'].replace(['bumi_malay','bumi_other'],'Bumiputera')
df_pop['ethnicity']=df_pop['ethnicity'].replace(['other_citizen','other_noncitizen'],'Others')
df_pop['ethnicity']=df_pop['ethnicity'].replace("chinese",'Chinese')
df_pop['ethnicity']=df_pop['ethnicity'].replace("indian",'Indian')
df_pop


# In[409]:


def long_run(data_b, data_a):
    diff = data_a.mean()-data_b.mean()
    #print(diff)
    # Perform t-test
    t_stat, p_value = ttest_ind(data_b, data_a)

    # Print results
    #print(f"T-Statistic: {t_stat}")
    #print(f"P-Value: {p_value}")
    #print('\n')
    
    lrd.append(diff)
    tstat.append(t_stat)
    pvalue.append(p_value)


# In[689]:


df_slr=pd.DataFrame()
df_slr['state']=df_state.columns
lrd=[]
tstat=[]
pvalue=[]

for state in df_state.columns: 
        #print(str(state))
        long_run(df_state.loc['2022-03-23':'2022-09-06'][state], df_state.loc['2022-10-08':'2023-03-23'][state])
        
df_slr['diff']=lrd
df_slr['tstat']=tstat
df_slr['pvalue']=pvalue
df_slr['%diff']=[df_slr['diff'][i] / df_sar['pop'][i] for i in range(len(pop))]
df_slr['%diff']=[df_slr['diff'][i] / df_sar['pop'][i] for i in range(len(pop))]
df_slr['IE_A']=[df_sar['shock1'][i] - df_slr['diff'][i] for i in range(len(pop))]
df_slr['IE_B']=[df_sar['shock2'][i] - df_slr['diff'][i] for i in range(len(pop))]
df_slr['%IE_A']=[df_sar['%shock1'][i] - df_slr['%diff'][i] for i in range(len(pop))]
df_slr['%IE_B']=[df_sar['%shock2'][i] - df_slr['%diff'][i] for i in range(len(pop))]
df_slr.set_index('state', inplace=True)
df_slr.to_latex('table10.tex')
df_slr


# In[688]:


df_elr=pd.DataFrame()
df_elr['eth']=df_eth.columns
lrd=[]
tstat=[]
pvalue=[]

for eth in df_eth.columns: 
        #print(str(eth))
        long_run(df_eth.loc['2022-03-23':'2022-09-06'][eth], df_eth.loc['2022-10-08':'2023-03-23'][eth])
        
df_elr['diff']=lrd
df_elr['tstat']=tstat
df_elr['pvalue']=pvalue
df_elr['%diff']=[df_elr['diff'][i] / df_ear['pop'][i] for i in range(len(epop))]
df_elr['%diff']=[df_elr['diff'][i] / df_ear['pop'][i] for i in range(len(epop))]
df_elr['IE_A']=[df_ear['shock1'][i] - df_elr['diff'][i] for i in range(len(epop))]
df_elr['IE_B']=[df_ear['shock2'][i] - df_elr['diff'][i] for i in range(len(epop))]
df_elr['%IE_A']=[df_ear['%shock1'][i] - df_elr['%diff'][i] for i in range(len(epop))]
df_elr['%IE_B']=[df_ear['%shock2'][i] - df_elr['%diff'][i] for i in range(len(epop))]
df_elr.set_index('eth', inplace=True)
df_elr.to_latex('table11.tex')
df_elr


# In[686]:


df_alr=pd.DataFrame()
df_alr['age']=df_age.columns
lrd=[]
tstat=[]
pvalue=[]

for age in df_age.columns: 
        #print(str(age))
        long_run(df_age.loc['2022-03-23':'2022-09-06'][age], df_age.loc['2022-10-08':'2023-03-23'][age])
        
df_alr['diff']=lrd
df_alr['tstat']=tstat
df_alr['pvalue']=pvalue
df_alr['%diff']=[df_alr['diff'][i] / df_aar['pop'][i] for i in range(len(apop))]
df_alr['IE_A']=[df_aar['shock1'][i] - df_alr['diff'][i] for i in range(len(apop))]
df_alr['IE_B']=[df_aar['shock2'][i] - df_alr['diff'][i] for i in range(len(apop))]
df_alr['%IE_A']=[df_aar['%shock1'][i] - df_alr['%diff'][i] for i in range(len(apop))]
df_alr['%IE_B']=[df_aar['%shock2'][i] - df_alr['%diff'][i] for i in range(len(apop))]
df_alr.set_index('age', inplace=True)
df_alr.to_latex('table12.tex')
df_alr


# In[761]:


df_apop = df_pop.groupby(['age', 'state'])['population'].sum().reset_index().pivot(index='age', columns='state', values='population')
df_apop.loc['total']=df_apop.sum(axis=0)
df_apop['Malaysia']=df_apop.sum(axis=1)
df_apop


# In[762]:


df_apop.drop('Malaysia', inplace=True, axis=1)
df_apop = df_apop.drop(df_apop.index[-1])
df_apop


# In[763]:


df_aar['pop']=[x/10 for x in apop]
df_aar['%shock1']=[df_aar['shock1'][i] / df_aar['pop'][i] for i in range(len(apop))]
df_aar['%shock2']=[df_aar['shock2'][i] / df_aar['pop'][i] for i in range(len(apop))]
#df_aar.set_index('age',inplace=True)


# In[764]:


for state in df_apop.columns:
    print(df_apop[state].sum())


# In[696]:


df_papop=pd.DataFrame()
for state in df_apop.columns:
    for index in df_apop.index:
        df_papop.loc[index, state] = df_apop.loc[index, state]/df_apop[state].sum()


# In[765]:


df_papop['index']=(df_aar['%shock1']+df_aar['%shock2'])/2*100
df_papop


# In[766]:


# sum values in column A and create a new row
age = []
for state in df_papop.columns:
    sum = 0
    for i in df_papop.index:
        sum += df_papop.loc[i, state]*df_papop.loc[i, 'index']
    age.append(sum)
age.pop()
age


# In[847]:


df_epop = df_pop.groupby(['ethnicity', 'state'])['population'].sum().reset_index().pivot(index='ethnicity', columns='state', values='population')
df_pepop=pd.DataFrame()
for state in df_epop.columns:
    for index in df_epop.index:
        df_pepop.loc[index, state] = df_epop.loc[index, state]/df_epop[state].sum()
df_pepop['index']=(df_ear['%shock1']+df_ear['%shock2'])/2*100
df_pepop


# In[848]:


eth = []
for state in df_pepop.columns:
    sum = 0
    for i in df_pepop.index:
        sum += df_pepop.loc[i, state]*df_pepop.loc[i, 'index']
    eth.append(sum)
eth.pop()
eth


# In[898]:


df_data = pd.read_csv('checkin_state.csv')
df_data['state'] = df_data['state'].replace(['Penang', 'Kuala Lumpur', 'Labuan', 'Putrajaya', 'Malacca'], ['Pulau Pinang', 'WP Kuala Lumpur', 'WP Labuan', 'WP Putrajaya', 'Melaka'])
df_data['age']=age
df_data['eth']=eth
df_data['dens']=np.log([209,225,119,583,180,44,118,348,1659,880,89,46,20,8157,1034,2215])
#df_data['%pop']=np.log(df_data['%pop'])
df_data['%pop']=[df_data['%pop'][i]/100 for i in range (len(df_state.columns))]
df_data['urb']=[0.774,0.673,0.441,0.909,0.693,0.528,0.72,0.538,0.925,0.958,0.642,0.547,0.57,1,0.889,1]
df_data.set_index('state', inplace=True)
df_data.drop(['pop'], inplace=True, axis=1)
df_data


# In[899]:





# In[ ]:





# In[900]:


np.sqrt(df_data.var(axis=0))/df_data.mean(axis=0)


# In[903]:


X = df_data[['%pop', 'internet', 'eth', 'age', 'vax', 'income_mean','urb']]
y = np.log((df_sar['%shock1'] + df_sar['%shock2']) / 2*100)
model = sm.OLS(y, sm.add_constant(X)).fit(cov_type='HAC', cov_kwds={'maxlags': 2})
model.summary()


# In[904]:


eigenvalues = np.linalg.eigvals(X.corr())
print(X.corr(), eigenvalues)
#vif = pd.DataFrame()
#vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#vif["features"] = X.columns
#vif


# In[901]:


X = df_data[['%pop', 'eth', 'age', 'vax', 'urb']]
y = np.log((df_sar['%shock1'] + df_sar['%shock2']) / 2*100)
model = sm.OLS(y, sm.add_constant(X)).fit(cov_type='HAC', cov_kwds={'maxlags': 2})
model.summary()


# In[908]:


y


# In[907]:


X = df_data[['eth', 'age', 'vax','urb']]
y = np.log((df_sar['%shock1'] + df_sar['%shock2']) / 2*100)
model = sm.OLS(y, sm.add_constant(X)).fit(cov_type='HC1')

# perform Breusch-Pagan test
bp_test = het_breuschpagan(model.resid, X)
print('Breusch-Pagan test p-value:', bp_test[0],bp_test[1])

# perform White test
X_clean = model.model.exog[~pd.isnull(model.model.endog).ravel(), :]
white_test = het_white(model.resid, X_clean)
print('White test p-value:', white_test[0],white_test[1])

latex_table = model.summary().as_latex()

# write the table to a file
with open('ols.tex', 'w') as f:
    f.write(latex_table)


# In[6]:


df_state


# In[7]:


df_state['Malaysia']=df_state.sum(axis=1)
df_state


# In[25]:


graph(df_state.loc['2022-03-23':'2022-09-06']['Malaysia'], df_state.loc['2022-09-07':'2023-03-23']['Malaysia'])


# In[26]:


graph2(df_state.loc['2022-03-23':'2022-09-30']['Malaysia'], df_state.loc['2022-10-01':'2023-03-23']['Malaysia'])


# In[ ]:





# In[ ]:




