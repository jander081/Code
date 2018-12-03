# -*- coding: utf-8 -*-

from collections import defaultdict
import pandas as pd
import numpy as np

from datetime import datetime

from faker import Faker
from collections import defaultdict


#------------------------------------------------------------


def make_dict(col, dframe):
    
    '''returns a dict for freqs. This can then be mapped to 
    any col to create freq feature. Must be run prior to freq_group'''
    
    
    df = pd.DataFrame(dframe[col].value_counts())
    df.reset_index(level=0, inplace=True)
    df.rename(columns={'index': 'key', col: 'value'}, inplace=True)
    df_dict = defaultdict(list)
    for k, v in zip(df.key, df.value):
        df_dict[k] = (int(v))
    return df_dict


#------------------------------------------------------------

def freq_group(freq, _dict, rare, infrequent, less_common):
    
    '''run as lambda on col'''
    
    rev_dict = {v:k for k, v in _dict.items()}
    
    if freq <= rare:
        string = 'rare'
    elif freq > rare and freq <= infrequent:
        string = 'infrequent'
    elif freq > infrequent and freq <= less_common:
        string = 'less common'
    else:
        string = rev_dict[freq]
    return(string)









 

#------------------------------------------------------------

def line_num_nonrenew_nb(dframe_curr, q_num_curr, dframe_prev, q_num_prev):  
    
    '''Quarterly agent number of non-renews and NB by line'''
    
    dframe_prev['Renew'] = dframe_prev['New-Renew Ind'].map({'New': 0, 'Renew': 0, 'Non-Renew': 1})
    
    # PIVOTs 
    # PREV QUARTER NON-RENEW
    prev_agnt_q_nonRenew_pv = pd.pivot_table(dframe_prev, values='Renew', index='agnt_', 
                                   columns = ['quarter_', 'line_'], aggfunc='sum').fillna(0) 
    # AUTO PREV  
    dict_agnt_prev_auto_nonRenew = pd.Series(prev_agnt_q_nonRenew_pv.loc[:, (q_num_prev, 'AUTO')], index=prev_agnt_q_nonRenew_pv.index).to_dict()
    dframe_curr.loc[:, 'prev_q_agnt_cmp_auto_nonRenew'] = dframe_curr.loc[:, 'agnt_'].map(dict_agnt_prev_auto_nonRenew) 
    
    # CMP PREV  
    dict_agnt_prev_cmp_nonRenew = pd.Series(prev_agnt_q_nonRenew_pv.loc[:, (q_num_prev, 'COMM MULTI-PERIL')], index=prev_agnt_q_nonRenew_pv.index).to_dict()
    dframe_curr.loc[:, 'prev_q_agnt_cmp_nonRenew'] = dframe_curr.loc[:, 'agnt_'].map(dict_agnt_prev_cmp_nonRenew) 
    
    # WC PREV  
    dict_agnt_prev_wc_nonRenew = pd.Series(prev_agnt_q_nonRenew_pv.loc[:, (q_num_prev, 'WORKERS COMP')], index=prev_agnt_q_nonRenew_pv.index).to_dict()
    dframe_curr.loc[:, 'prev_q_agnt_wc_nonRenew'] = dframe_curr.loc[:, 'agnt_'].map(dict_agnt_prev_wc_nonRenew) 
    
    # NEW BUSINESS
    dframe_prev['NB'] = dframe_prev['New-Renew Ind'].map({'New': 1, 'Renew': 0, 'Non-Renew': 0})
    dframe_curr['NB'] = dframe_curr['New-Renew Ind'].map({'New': 1, 'Renew': 0, 'Non-Renew': 0})
    
    # PIVOTs 
    # CURR QUARTER NB
    curr_agnt_q_nb_pv = pd.pivot_table(dframe_curr, values='NB', index='agnt_', 
                                   columns = ['quarter_', 'line_'], aggfunc='sum').fillna(0)     
    # PREV QUARTER NB
    prev_agnt_q_nb_pv = pd.pivot_table(dframe_prev, values='NB', index='agnt_', 
                                   columns = ['quarter_', 'line_'], aggfunc='sum').fillna(0) 
    # AUTO NB CURRENT
    dict_agnt_curr_nb_auto = pd.Series(curr_agnt_q_nb_pv.loc[:, (q_num_curr, 'AUTO')], index=curr_agnt_q_nb_pv.index).to_dict()    
    dframe_curr.loc[:, 'curr_q_agnt_auto_nb'] = dframe_curr.loc[:, 'agnt_'].map(dict_agnt_curr_nb_auto)
    # PREV      
    dict_agnt_prev_nb_auto = pd.Series(prev_agnt_q_nb_pv.loc[:, (q_num_prev, 'AUTO')], index=prev_agnt_q_nb_pv.index).to_dict()
    dframe_curr.loc[:, 'prev_q_agnt_auto_nb'] = dframe_curr.loc[:, 'agnt_'].map(dict_agnt_prev_nb_auto) 
       
    
    # CMP CURRENT
    dict_agnt_curr_nb_cmp = pd.Series(curr_agnt_q_nb_pv.loc[:, (q_num_curr, 'COMM MULTI-PERIL')], index=curr_agnt_q_nb_pv.index).to_dict()    
    dframe_curr.loc[:, 'curr_q_agnt_cmp_nb'] = dframe_curr.loc[:, 'agnt_'].map(dict_agnt_curr_nb_cmp)
    # PREV      
    dict_agnt_prev_nb_cmp = pd.Series(prev_agnt_q_nb_pv.loc[:, (q_num_prev, 'COMM MULTI-PERIL')], index=prev_agnt_q_nb_pv.index).to_dict()
    dframe_curr.loc[:, 'prev_q_agnt_cmp_nb'] = dframe_curr.loc[:, 'agnt_'].map(dict_agnt_prev_nb_cmp) 
       
    # WC CURRENT
    dict_agnt_curr_nb_wc = pd.Series(curr_agnt_q_nb_pv.loc[:, (q_num_curr, 'WORKERS COMP')], index=curr_agnt_q_nb_pv.index).to_dict()    
    dframe_curr.loc[:, 'curr_q_agnt_wc_nb'] = dframe_curr.loc[:, 'agnt_'].map(dict_agnt_curr_nb_wc)
    # PREV      
    dict_agnt_prev_nb_wc = pd.Series(prev_agnt_q_nb_pv.loc[:, (q_num_prev, 'WORKERS COMP')], index=prev_agnt_q_nb_pv.index).to_dict()
    dframe_curr.loc[:, 'prev_q_agnt_wc_nb'] = dframe_curr.loc[:, 'agnt_'].map(dict_agnt_prev_nb_wc) 
               
    dframe_prev.drop(['NB', 'Renew'], axis=1, inplace=True)
    
    dframe_curr.drop(['NB'], axis=1, inplace=True)
    
    dframe_curr = dframe_curr.fillna(0)
    
    return(dframe_curr)

#------------------------------------------------------------

def acc_feat(dframe_curr, q_num_curr, dframe_prev, q_num_prev, period):
    
    '''returns a dataframe with all new features'''
    
    '''returns the pol_life as of the specified period, which is meant to be quarterly.
    new policies within the quarter will be <= 91. This returns the entire dataframe with the
    added feature. Should not be run for q1 2016.'''
    
    # POL_LIFE
    ten_feat = (pd.concat([dframe_curr['New-Renew Ind'], 
                        dframe_curr['Effective YearMonth'].apply(str), 
                        dframe_curr['Line Tenure'].apply(str)], axis=1)).apply(lambda x: ' '.join(x), axis=1)
    
    ten = pd.DataFrame(ten_feat.apply(lambda x: NB_adj_tenure(x, period))).rename(columns={0:'tenure'})
    
    df_ = pd.concat([dframe_curr, ten], axis=1) 
    
    pol_feat = pd.concat([df_['New-Renew Ind'], 
                        df_['Effective YearMonth'].apply(str), df_['tenure']],
                        axis=1).apply(lambda x: ' '.join(x), axis=1)
    
    pol_feat = pol_feat.apply(lambda x: delta(x, period))
    
    dframe_curr['pol_life'] = pd.DataFrame(pol_feat).rename(columns={0:'pol_life'})
    
    '''Functions are located above. Splits up by line'''
    
    dframe_curr = line_tot_prem(dframe_curr, q_num_curr, dframe_prev, q_num_prev)
        
#     # PREVIOUS QUARTER AGENT TOTAL PREM 
#     p_agt_prev = pd.pivot_table(dframe_prev, values='base', index='agent', 
#                                    columns = 'quarter', aggfunc='sum').fillna(0)   
#     dict_agt_prev = pd.Series(p_agt_prev[q_num_prev], index=p_agt_prev.index).to_dict()
#     dframe_curr['prev_q_agt_tot_prem'] = dframe_curr['agent'].map(dict_agt_prev) 
    
#     # CURRENT QUARTER AGENT TOTAL PREM 
#     p_agt_curr = pd.pivot_table(dframe_curr, values='base', index='agent', 
#                                    columns = 'quarter', aggfunc='sum').fillna(0)   
#     dict_agt_curr = pd.Series(p_agt_curr[q_num_curr], index=p_agt_curr.index).to_dict()    
#     dframe_curr['curr_q_agt_tot_prem'] = dframe_curr['agent'].map(dict_agt_curr)
    
    dframe_curr = line_num_acc(dframe_curr, q_num_curr, dframe_prev, q_num_prev)
    
    dframe_curr = line_num_nonrenew_nb(dframe_curr, q_num_curr, dframe_prev, q_num_prev)


#     # PREVIOUS QUARTER AGENT NUMBER OF ACCOUNTS
#     agt_dict_prev = make_dict('agent', dframe_prev)
#     dframe_curr['prev_q_agt_num_acc'] = dframe_curr['agent'].map(agt_dict_prev)

#     # CURRENT QUARTER AGENT NUMBER OF ACCOUNTS
#     agt_dict_curr = make_dict('agent', dframe_curr)
#     dframe_curr['curr_q_agt_num_acc'] = dframe_curr['agent'].map(agt_dict_curr)

            
    # CURRENT QUARTER CLIENT TOTAL PREM 
    curr_clt = pd.pivot_table(dframe_curr, values='prem_', index='pol_num', 
                                   columns = 'quarter_', aggfunc='sum').fillna(0)
    dict_clt = pd.Series(curr_clt[q_num_curr], index=curr_clt.index).to_dict()
    dframe_curr.loc[:, 'curr_clt_tot_prem'] = dframe_curr.loc[:, 'pol_num'].map(dict_clt)
    
    # PREVIOUS QUARTER CLIENT TOTAL PREM 
    prev_clt = pd.pivot_table(dframe_prev, values='prem_', index='pol_num', 
                                   columns = 'quarter_', aggfunc='sum').fillna(0)
    dict_clt_prev = pd.Series(prev_clt[q_num_prev], index=prev_clt.index).to_dict()
    dframe_curr.loc[:, 'prev_clt_tot_prem'] = dframe_curr.loc[:, 'pol_num'].map(dict_clt_prev)
    
    
    # CURRENT QUARTER CLIENT NUMBER OF ACCOUNTS
    pol_num_dict = make_dict('pol_num', dframe_curr)
    dframe_curr.loc[:, 'curr_clt_num_acc'] = dframe_curr.loc[:, 'pol_num'].map(pol_num_dict)  
    
    
     # PREVIOUS QUARTER CLIENT NUMBER OF ACCOUNTS
    pol_num_dict_prev = make_dict('pol_num', dframe_prev)
    dframe_curr.loc[:, 'prev_clt_num_acc'] = dframe_curr.loc[:, 'pol_num'].map(pol_num_dict_prev)              
      
    # PREVIOUS QUARTER MSTR AGENT NUMBER OF ACCOUNTS
    mstr_dict_prev = make_dict('mstr_agnt', dframe_prev)
    dframe_curr.loc[:, 'prev_q_mstr_agnt_num_acc'] = dframe_curr.loc[:, 'mstr_agnt'].map(mstr_dict_prev)
    
    # CURRENT QUARTER MSTR AGENT NUMBER OF ACCOUNTS
    mstr_dict_curr = make_dict('mstr_agnt', dframe_curr)
    dframe_curr.loc[:, 'curr_q_mstr_agnt_num_acc'] = dframe_curr.loc[:, 'mstr_agnt'].map(mstr_dict_curr)
    
    # PREVIOUS QUARTER PROGRAM NUM OF ACCOUNTS
    prgm_dict_prev = make_dict('prgm_', dframe_prev)
    dframe_curr.loc[:, 'prev_q_pgrm_num_acc'] = dframe_curr.loc[:, 'prgm_'].map(prgm_dict_prev)
    
    # CURRENT QUARTER PROGRAM NUM OF ACCOUNTS
    prgm_dict_curr = make_dict('prgm_', dframe_curr)
    dframe_curr.loc[:, 'curr_q_pgrm_num_acc'] = dframe_curr.loc[:, 'prgm_'].map(prgm_dict_curr)   
    
    
    return(dframe_curr)
    

#------------------------------------------------------------

def equal_feat(dframe, period):
    
    # UW FREQ
    UW_dict = make_dict('UW_', dframe)
    dframe['UW_freq'] = np.float64(dframe['UW_'].map(UW_dict))
    
    # UW TITLES
    fake = Faker()
    UW_names  = defaultdict(fake.last_name) 
    dframe['UW_title'] = dframe['UW_'].apply(lambda x: UW_names[x])
    
    dframe.drop(['UW_'], axis=1, inplace=True)
    
    # SE FREQ
    SE_dict = make_dict('Sales Executive Pd', dframe)
    dframe['SE_freq'] = np.float64(dframe['Sales Executive Pd'].map(SE_dict))

    # SE TITLES
    fake = Faker()
    SE_names  = defaultdict(fake.last_name) 
    dframe['SE_title'] = dframe['Sales Executive Pd'].apply(lambda x: SE_names[x])
    
    dframe.drop(['Sales Executive Pd'], axis=1, inplace=True)
    
    # AG FREQ - USE TO REDUCE AGENT VALUES
    AG_dict = make_dict('agnt_', dframe)
    dframe['agnt_freq'] = np.float64(dframe['agnt_'].map(AG_dict))
    
    # AG REVISED - NOT ANONYMIZED
    dframe['agnt_rev'] = dframe['agnt_freq'].apply(lambda x: freq_group(x, AG_dict, 1, 5, 10))
    
    # AG TITLES ANONYMIZE
    fake = Faker()
    AG_names  = defaultdict(fake.company) 
    dframe['agnt_'] = dframe['agnt_'].apply(lambda x: AG_names[x])
    
    # MSTR FREQ - USE TO REDUCE AGENT VALUES
    MSTR_dict = make_dict('mstr_agnt', dframe)
    dframe['mstr_freq'] = np.float64(dframe['mstr_agnt'].map(MSTR_dict))
    
    #MSTR Revised - NOT ANONYMIZED
    dframe['mstr_rev'] = dframe['mstr_freq'].apply(lambda x: freq_group(x, MSTR_dict, 10, 50, 100))
    
    # MSTR TITLES ANONYMIZE
    fake = Faker()
    MSTR_names  = defaultdict(fake.company) 
    dframe['mstr_agnt'] = dframe['mstr_agnt'].apply(lambda x: MSTR_names[x])
   
    # ZIP
    zip_dict = make_dict('zip_code', dframe)
    dframe['zip_freq'] = np.float64(dframe['zip_code'].map(zip_dict))

    #ZIP ANONYMIZE
    fake = Faker()
    zip_names  = defaultdict(fake.zipcode)
    dframe['zip_num'] = dframe['zip_code'].apply(lambda x: zip_names[x])
    dframe.drop(['zip_code'], axis=1, inplace=True)
    
    # COUNTY
    county_dict = make_dict('county_', dframe)
    dframe['county_freq'] = np.float64(dframe['county_'].map(county_dict))
    
    # PREM FREQ
    
    prem_dict = make_dict('prem_bin', dframe)
    dframe['prem_freq'] = np.float64(dframe['prem_bin'].map(prem_dict))
    
    
    # TERRITORY
    # with the state, subdivided territory < 30 
    # REALLY CAN'T USE THIS. THE VALUES ARE TOO HIGH AND NOT ASSIGNED TO A PARTICULAR COUNTY
    

    return(dframe)
    
    
    



#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

















       