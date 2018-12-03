# -*- coding: utf-8 -*-

from collections import defaultdict
import pandas as pd
import numpy as np

from datetime import datetime

from faker import Faker
from collections import defaultdict


#------------------------------------------------------------

def df_quarters_set(dframe):
    
# since youre appending to a list, df_1 will be equal to dataset[0]. 
# So the holy crap code makes sense - for now

    df_lst = []
    for i in range(1, 11):
        
        sub = dframe[dframe['abs_Q'] == i]
        sub.name = 'df_' + str(i)
        
        df_lst.append(sub)
    return(df_lst)   


#------------------------------------------------------------

def abs_Q(dframe):

    dframe['term'] = dframe['year_'] + '_' + dframe['quarter_']

    abs_quarter = { '2016_1': 1, '2016_2': 2, '2016_3': 3 ,'2016_4': 4,
                   '2017_1': 5, '2017_2': 6, '2017_3': 7,'2017_4': 8,'2018_1': 9, '2018_2': 10}

    dframe['abs_Q'] = dframe['term'].map(abs_quarter)

    return(dframe)

#------------------------------------------------------------

def prem_lag(dframe, entity, col, quarter_num, year, base):
    
    df_feat = dframe[[entity]]
    
    for j in range(1, quarter_num + 1):
    
        sub = dframe[(dframe['quarter_'] == str(j)) & (dframe['year_'] == str(year)) &\
                     (dframe['New-Renew Ind'] != 'Non-Renew')]

        base_red = base[ -( base['quarter_'] == str(j) ) ]
        base_update = base_red.append(sub)
        
        
        pv = pd.pivot_table(base_update, values='prem_', index=entity, 
                                           columns = [col], aggfunc='sum').fillna(0)


        for i in np.unique(dframe[col]):

            dict_ = pd.Series(pv.loc[:, i], index=pv.index).to_dict() 

            feat_name = entity[:2] + '_' + str(i).lower()[:1] + 'Q' + str(j) + '_' + str(year)[2:]

            df_feat[feat_name] = dframe[entity].map(dict_)

    
    return(df_feat.set_index(entity))

#------------------------------------------------------------


def holy_crap(dframe, datasets):
    
    '''see above for dataset explanation. This needs to be restructured and organized. The temptation is to 
    start pumping out new features. not a good idea. Test the initial lag features and incorporate them into your
    work. In your spare time, disect and re-write this code. It takes visuals to understand this process.'''
    
       
    base_18 = dframe[(dframe['New-Renew Ind'] != 'Non-Renew') & (dframe['year_'] == '2017')] # Q4 2017
    base_17 = dframe[(dframe['New-Renew Ind'] != 'Non-Renew') & (dframe['year_'] == '2016')] # Q4 2016
    base_16 = dframe[(dframe['New-Renew Ind'] != 'New') & (dframe['year_'] == '2016')] # Q4 2015

    feats_18 = prem_lag(dframe, 'agnt_', 'line_', 2, 2018, base_18)
    feats_17 = prem_lag(dframe, 'agnt_', 'line_', 4, 2017, base_17)
    feats_16 = prem_lag(dframe, 'agnt_', 'line_', 4, 2016, base_16)
        
    feats_merge = pd.concat([feats_16, feats_17, feats_18], axis=1)
        
    # QUARTER 2
    n = 2
    df_2 = datasets[n-1] # data
    # admittedly, the dataset indexing is lousy. It was meant to facilitate streamlined coding, but
    # I ran out of time. This code needs a lot more love.
    
    # first lag. For Q2 2016. lags are in groups of 3 lines
    dict_a = pd.Series(feats_merge.iloc[:, 0], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 1], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 2], index=feats_merge.index).to_dict()
    

    # the dict is grabbing the previous quarter info
    df_2['lag_1_auto'] = datasets[1]['agnt_'].map(dict_a)
    df_2['lag_1_cmp'] = datasets[1]['agnt_'].map(dict_c)
    df_2['lag_1_wc'] = datasets[1]['agnt_'].map(dict_w)
    

    # LAG 2 REQUIRES SPECIAL ATTENTION FOR THIS QUARTER ONLY
    # This quarter represents 4Q 2015 covering the entire 2015 year and is based off 2016 existing policies (renew and non) 
    # minus NB
        
    # Its a strong assumption, but it is need to use 2016. This approximated year updates every quarter until it becomes
    # base 2017 in 4Q 
    base_pv = pd.pivot_table(base_16, values='prem_', index='agnt_', 
                                       columns = ['line_'], aggfunc='sum').fillna(0)
  
    # Notice the pivot table uses base_pv
    dict_base_a = pd.Series(base_pv.iloc[:, 0], index=base_pv.index).to_dict() 
    dict_base_c = pd.Series(base_pv.iloc[:, 1], index=base_pv.index).to_dict() 
    dict_base_w = pd.Series(base_pv.iloc[:, 2], index=base_pv.index).to_dict()    

    df_2['lag_2_auto'] = datasets[1]['agnt_'].map(dict_base_a)
    df_2['lag_2_cmp'] = datasets[1]['agnt_'].map(dict_base_c)
    df_2['lag_2_wc'] = datasets[1]['agnt_'].map(dict_base_w)
   
    # QUARTER 3
    entity = 'agnt_'
    n = 3
    df_3 = datasets[n-1]
    
    #lag 1 - Q2
    dict_a = pd.Series(feats_merge.iloc[:, 3], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 4], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 5], index=feats_merge.index).to_dict() 
#     dict_pol = pd.Series(feats_merge_pol.iloc[:, 0], index=feats_merge_pol.index).to_dict()

    df_3['lag_1_auto'] = datasets[n-1][entity].map(dict_a)
    df_3['lag_1_cmp'] = datasets[n-1][entity].map(dict_c)
    df_3['lag_1_wc'] = datasets[n-1][entity].map(dict_w)
    
    #lag 2 -Q3
    dict_a = pd.Series(feats_merge.iloc[:, 0], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 1], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 2], index=feats_merge.index).to_dict() 

    df_3['lag_2_auto'] = datasets[n-1][entity].map(dict_a)
    df_3['lag_2_cmp'] = datasets[n-1][entity].map(dict_c)
    df_3['lag_2_wc'] = datasets[n-1][entity].map(dict_w)
    
    # QUARTER 4
    n=4
    df_4 = datasets[n-1]
    
    #lag 1 - Q3
    dict_a = pd.Series(feats_merge.iloc[:, 6], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 7], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 8], index=feats_merge.index).to_dict() 

    df_4['lag_1_auto'] = datasets[n-1][entity].map(dict_a)
    df_4['lag_1_cmp'] = datasets[n-1][entity].map(dict_c)
    df_4['lag_1_wc'] = datasets[n-1][entity].map(dict_w)
    
    #lag 2 - Q2
    dict_a = pd.Series(feats_merge.iloc[:, 3], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 4], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 5], index=feats_merge.index).to_dict() 

    df_4['lag_2_auto'] = datasets[n-1][entity].map(dict_a)
    df_4['lag_2_cmp'] = datasets[n-1][entity].map(dict_c)
    df_4['lag_2_wc'] = datasets[n-1][entity].map(dict_w)
    

    # QUARTER 5
    n=5
    df_5 = datasets[n-1]
    
    #lag 1 - Q4
    dict_a = pd.Series(feats_merge.iloc[:, 9], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 10], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 11], index=feats_merge.index).to_dict() 

    df_5['lag_1_auto'] = datasets[n-1][entity].map(dict_a)
    df_5['lag_1_cmp'] = datasets[n-1][entity].map(dict_c)
    df_5['lag_1_wc'] = datasets[n-1][entity].map(dict_w)
    
    #lag 2 - Q3
    dict_a = pd.Series(feats_merge.iloc[:, 6], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 7], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 8], index=feats_merge.index).to_dict() 

    df_5['lag_2_auto'] = datasets[n-1][entity].map(dict_a)
    df_5['lag_2_cmp'] = datasets[n-1][entity].map(dict_c)
    df_5['lag_2_wc'] = datasets[n-1][entity].map(dict_w)
    
    
    # QUARTER 6
    n=6
    df_6 = datasets[n-1]
    
    #lag 1 - Q5
    # start of the second run of the algorithm. Base 17 updated to create Q5
    dict_a = pd.Series(feats_merge.iloc[:, 12], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 13], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 14], index=feats_merge.index).to_dict() 

    df_6['lag_1_auto'] = datasets[n-1][entity].map(dict_a)
    df_6['lag_1_cmp'] = datasets[n-1][entity].map(dict_c)
    df_6['lag_1_wc'] = datasets[n-1][entity].map(dict_w)

    #lag 2 - Q6
    dict_a = pd.Series(feats_merge.iloc[:, 9], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 10], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 11], index=feats_merge.index).to_dict() 

    df_6['lag_2_auto'] = datasets[n-1][entity].map(dict_a)
    df_6['lag_2_cmp'] = datasets[n-1][entity].map(dict_c)
    df_6['lag_2_wc'] = datasets[n-1][entity].map(dict_w)
        
    
    # QUARTER 7
    n=7
    df_7 = datasets[n-1]    
    
    #lag 1 - Q6
    dict_a = pd.Series(feats_merge.iloc[:, 15], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 16], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 17], index=feats_merge.index).to_dict() 

    df_7['lag_1_auto'] = datasets[n-1][entity].map(dict_a)
    df_7['lag_1_cmp'] = datasets[n-1][entity].map(dict_c)
    df_7['lag_1_wc'] = datasets[n-1][entity].map(dict_w)

    #lag 2 - Q5
    dict_a = pd.Series(feats_merge.iloc[:, 12], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 13], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 14], index=feats_merge.index).to_dict() 

    df_7['lag_2_auto'] = datasets[n-1][entity].map(dict_a)
    df_7['lag_2_cmp'] = datasets[n-1][entity].map(dict_c)
    df_7['lag_2_wc'] = datasets[n-1][entity].map(dict_w)
    
    
    # QUARTER 8
    n=8
    df_8 = datasets[n-1] 

    #lag 1 - Q7
    dict_a = pd.Series(feats_merge.iloc[:, 18], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 19], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 20], index=feats_merge.index).to_dict() 

    df_8['lag_1_auto'] = datasets[n-1][entity].map(dict_a)
    df_8['lag_1_cmp'] = datasets[n-1][entity].map(dict_c)
    df_8['lag_1_wc'] = datasets[n-1][entity].map(dict_w)
    
    #lag 2 - Q6
    dict_a = pd.Series(feats_merge.iloc[:, 15], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 16], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 17], index=feats_merge.index).to_dict() 

    df_8['lag_2_auto'] = datasets[n-1][entity].map(dict_a)
    df_8['lag_2_cmp'] = datasets[n-1][entity].map(dict_c)
    df_8['lag_2_wc'] = datasets[n-1][entity].map(dict_w)
    
      
    # QUARTER 9
    n=9
    df_9 = datasets[n-1] 
    
    #lag 1 - Q8
    dict_a = pd.Series(feats_merge.iloc[:, 21], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 22], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 23], index=feats_merge.index).to_dict() 

    df_9['lag_1_auto'] = datasets[n-1][entity].map(dict_a)
    df_9['lag_1_cmp'] = datasets[n-1][entity].map(dict_c)
    df_9['lag_1_wc'] = datasets[n-1][entity].map(dict_w)

    #lag 2 - Q7
    dict_a = pd.Series(feats_merge.iloc[:, 18], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 19], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 20], index=feats_merge.index).to_dict() 

    df_9['lag_2_auto'] = datasets[n-1][entity].map(dict_a)
    df_9['lag_2_cmp'] = datasets[n-1][entity].map(dict_c)
    df_9['lag_2_wc'] = datasets[n-1][entity].map(dict_w)
    
    # QUARTER 10
    n=10
    df_10 = datasets[n-1] 
    
    #lag 1 - Q9
    # start of the third run of the algorithm. Base 18 updated to create Q9
    dict_a = pd.Series(feats_merge.iloc[:, 24], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 25], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 26], index=feats_merge.index).to_dict() 

    df_10['lag_1_auto'] = datasets[n-1][entity].map(dict_a)
    df_10['lag_1_cmp'] = datasets[n-1][entity].map(dict_c)
    df_10['lag_1_wc'] = datasets[n-1][entity].map(dict_w)

    #lag 2 - Q8
    dict_a = pd.Series(feats_merge.iloc[:, 21], index=feats_merge.index).to_dict() 
    dict_c = pd.Series(feats_merge.iloc[:, 22], index=feats_merge.index).to_dict() 
    dict_w = pd.Series(feats_merge.iloc[:, 23], index=feats_merge.index).to_dict() 

    df_10['lag_2_auto'] = datasets[n-1][entity].map(dict_a)
    df_10['lag_2_cmp'] = datasets[n-1][entity].map(dict_c)
    df_10['lag_2_wc'] = datasets[n-1][entity].map(dict_w)
    
    
    df_final = pd.concat([df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10], axis=0)
    
    return(df_final)
    
    
    
    

#------------------------------------------------------------
#------------------------------------------------------------

















       