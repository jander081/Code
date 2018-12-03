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
def csf(state):
    
    '''maps for competitive state fund'''
    
    csf = ['AZ', 'CA', 'CO', 
       'HI', 'ID', 'KY', 'LA', 'ME',
       'MD', 'MN', 'MO', 'MT', 'NM', 
       'NY', 'OK', 'OR', 'PA', 'RI', 'TX', 'UT'] 

    if state in csf:
        state = '1'
    else:
        state = '0'
    return(state)    

#------------------------------------------------------------

#NEW FEATURES
'''TECH dictionary for mapping'''

tech_index = {'AK': 44.86, 'AZ': 54.88, 'CO': 80.40, 'ID': 46.30, 'MT': 43.73,
          'NM': 55.19,'NV': 32.76, 'OR': 62.33, 'UT': 69.14, 'WA': 71.84, 
          'WY': 43.02, 
          
          'IL': 59.51, 'IN': 49.23, 'IA': 43.52, 'KY': 30.53, 
          'MI': 58.75, 'MN': 69.58, 'NE': 53.53, 'ND': 49.73, 'OH': 52.32, 
          'SD': 41.55, 'WI': 55.06,
          
          'AL': 42.67, 'FL': 38.82, 'GA': 53.53, 
          'MS': 29.84, 'NC': 62.64, 'SC': 35.84, 'TN': 40.22,
          
          'CT': 71.05, 'MA': 83.67,  'ME': 38.39,  'NH': 65.32,  
          'NY': 57.55,  'RI': 59.84,  'VT': 52.58, 
         
          'DC': 72.00, 'DE': 65.38, 'MD': 80.31, 'NJ': 59.40, 
          'PA': 61.54, 'VA': 65.88, 'WV': 25.84, 'CA':75.94, 
         
          'AR': 27.95, 'KS': 48.44, 'LA': 31.40, 'MO': 50.60, 
          'OK': 34.62, 'TX': 58.66}

#---------------------------------------------------------------------  

#NEW FEATURES
'''Education dictionary for mapping'''

ed_index = {'AK': 27.9, 'AZ': 25.9, 'CO': 36.4, 'ID': 24.4, 'MT': 28.8,
          'NM': 25.0,'NV': 21.7, 'OR': 28.8, 'UT': 29.3, 'WA': 31.1, 
          'WY': 24.1, 
          
          'IL': 30.8, 'IN': 22.7, 'IA': 24.9, 'KY': 20.5, 
          'MI': 25.2, 'MN': 31.8, 'NE': 28.6, 'ND': 27.6, 'OH': 24.6, 
          'SD': 26.3, 'WI': 26.3,
          
          'AL': 21.90, 'FL': 25.8, 'GA': 27.3, 
          'MS': 19.5, 'NC': 26.5, 'SC': 24.5, 'TN': 23.1,
          
          'CT': 35.5, 'MA': 39.0,  'ME': 26.8,  'NH': 32.8,  
          'NY': 32.5,  'RI': 30.2,  'VT': 33.6, 'CA': 30.1,
         
          'DC': 50.1, 'DE': 27.8, 'MD': 36.1, 'NJ': 35.4, 
          'PA': 27.1, 'VA': 34.2, 'WV': 17.5,
         
          'AR': 19.5, 'KS': 29.8, 'LA': 21.4, 'MO': 25.6, 
          'OK': 22.9, 'TX': 25.9}


#---------------------------------------------------------------------  
def market_sh(state):
    
    '''market share for states with competitve state funds'''
    
    high = ['CO', 'ID', 'MT', 'RI', 'ME', 'OR', 'UT'] #  > 50%
    med = ['HI', 'KY', 'NM', 'NY', 'OK', 'TX']  # > 25%
    low = ['CA', 'MN', 'PA', 'AZ', 'LA', 'MD', 'MO']  # > 0
    
    if state in high:
        state = 'high'
    elif state in med:
        state = 'med'
    elif state in low:
        state = 'low'
    else:
        state = 'none'
    return(state)

#---------------------------------------------------------------------   


def div(state):
    
    '''Div buckets for states with competitive state funds'''
    
    high = ['LA', 'MT', 'OR', 'TX'] #  > 18%
    med = ['CO', 'ME', 'NY', 'UT']  # > 5%
    low = ['AZ', 'CA', 'HI', 'ID', 'MD', 'MN', 'MO', 'RI']  # > 0
    
    if state in high:
        state = 'high'
    elif state in med:
        state = 'med'
    elif state in low:
        state = 'low'
    else:
        state = 'none'
    return(state)

#---------------------------------------------------------------------  

def invar_feat(dframe):

    # URBANITY
    FP_zip = pd.read_csv('/Users/jacob/Desktop/studies/misc/Thesis/churn_model/data/raw/ZIP-COUNTY-FIPS_2017-06.csv')

    urban = pd.read_csv('/Users/jacob/Desktop/studies/misc/Thesis/churn_model/data/raw/urbanity.csv')
    

    # THE TYPICAL METHOD FAILED DUE TO DUPLICATE FIPS. COULD PROBABLY MAKE IT WORK, BUT THE BELOW LAMBDA IS EASIER
    makeDict = lambda keys, values: {k:v for k, v in zip(keys, values)}
    zip_dict = makeDict(FP_zip.ZIP, FP_zip.STCOUNTYFP)
    dframe['FIPS'] = np.int64(dframe['zip_code'].map(zip_dict))

    UI_dict = makeDict(urban.FIPS, urban.UIC_2013)
    dframe['UI_code'] = np.float64(dframe['FIPS'].map(UI_dict))

    dframe.drop(['FIPS'], axis=1, inplace=True)
    
    
    
    # TRAVELERS EXPRESS - RENAMED
    dframe['Travelers Express Ind'].fillna('legacy', inplace=True)
    # RENAME
    dframe['Travelers Express Ind'] = dframe['Travelers Express Ind'].map({'Travelers Express':'express', 'SAMA':'legacy', 'legacy':'legacy'})
    
    dframe['price_complex_ind'] = dframe['Travelers Express Ind'].map({'express':1, 'legacy':0})
    dframe.drop(['Travelers Express Ind'], axis=1, inplace=True)
   
    # CON-TYPE
    dframe['con_type'].fillna('other', inplace=True)

    # PROGRAM - RENAMED
    dframe['prgm_'] = dframe['Program Pd'].apply(lambda x: x.replace('PAC', 'omit1').replace('PPLUS', 'omit2'))
    dframe.drop(['Program Pd'], axis=1, inplace=True)

    
    # DIFFERENT MSTR AGNT INDICATOR
    dframe['diff_mst_agnt'] = pd.np.where(dframe['mstr_agnt'] != dframe['agnt_'] , 1, 0)

    # DIFFERENT STATE INDICATOR
    dframe['diff_state'] = pd.np.where(dframe['Agent State Cd'] != dframe['risk_state'] , 1, 0)
    dframe.drop(['Agent State Cd'], axis=1, inplace=True)

    dframe['csf_ind'] = dframe['risk_state'].map(csf)
   

    dframe['csf_market_sh'] = dframe['risk_state'].map(market_sh)
    dframe['csf_market_sh'] = np.float64(dframe['csf_market_sh'].map({'none':0, 'low':1, 'med':2, 'high':3}))
    


    dframe['csf_div'] = dframe['risk_state'].map(div)
    dframe['csf_div'] = np.float64(dframe['csf_div'].map({'none':0, 'low':1, 'med':2, 'high':3}))
    
    # Growth should NOT be numeric
    # RENAME
#     dframe['growth'] = dframe['growth'].map({'Superior':4 , 'Balanced':3 , 'Property':2, 
#                                      'Selective':1, 
#                                      'Not A Market':0})
    

    dframe['tech_score'] = dframe['risk_state'].map(tech_index)

    dframe['ed_score'] = dframe['risk_state'].map(ed_index)

    # MONOMULTI 
    dframe['M_ind'] = dframe['M_ind'].map({'MONO': 0, 'MULTI': 1})
    

    #COAST -> 
    dframe['coast_ind'] = dframe['coast_ind'].map({'N': 0, 'Y': 1})
    

    #PROCESSING CENTER
    # convert to ordinal 1 - 3 amount of personal interaction
    dframe['PC_num'] = np.float64(dframe['Processing Center'].map({'USC':1, 'Service Center':2, 'Field':3}))
    dframe['PC_cat'] = dframe['Processing Center']
    dframe.drop(['Processing Center'], axis=1, inplace=True)

    
    dframe['risk_complex_ind'] = dframe['risk_complex_ind'].map({'Express': 0, 'Express Plus': 1})
 

    # SALES EX
    # RENAME
    dframe['sal_serv_ind'] = dframe['Inside Sales Executive Ind Pd'].map({'O':1, 'I':0})
    dframe.drop(['Inside Sales Executive Ind Pd'], axis=1, inplace=True)

    
    print(dframe.shape)
    return(dframe)

#------------------------------------------------------------

def NB_adj_tenure(string, period):
    
    '''this needs to be run once you determine the period to cover. So the tenure
    will change depending on the time period you are evaluating.'''
    
    churn = string.split()[0]
    tenure = string.split()[2]
    eff = datetime.strptime(string.split()[1],'%Y%m')
    if eff >  period.replace(year=int(period.year - 1)) and churn == 'New':
        ten = '0'
    elif eff ==  period.replace(year=int(period.year - 1)) and churn == 'New':
        ten = '1'
    elif eff <  period.replace(year=int(period.year - 1)) and churn == 'New':
        ten = 'remove'
    elif tenure == 'nan':
        ten = 'remove'       
    else:    
        ten = tenure
    return(ten)

#------------------------------------------------------------

def delta(string, period):
    
    '''Greatly simplified'''
    eff = datetime.strptime(string.split()[1],'%Y%m')
    tenure = int(float(string.split()[2]))
    if string.split()[0] == 'Non-Renew':
        policy_life = 365*tenure
        
    elif string.split()[0] == 'New':
        delta = period - eff
        policy_life = delta.days 
        
    else:
        
        delta = period - eff
        policy_life = np.float64(delta.days + 365*tenure)
    return(policy_life)




#------------------------------------------------------------

def quin(string):
    '''If the account is New, the Quintile is taken from the Quintile column - which
    has more values for New. Otherwise, it comes from Quintile at Renewal - which has
    more values for existing accounts'''
    
    churn = string.split()[0]
    q1 =string.split()[1]
    q2 = string.split()[2]
    if churn == 'New':
        quintile = q2
    else:
        quintile = q1
    if quintile == 'nan':
        quintile = quintile
    else:
        quintile = int(float(quintile))
        
    return(quintile)

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
    dframe_curr.loc[:, 'prev_q_agnt_auto_nonRenew'] = dframe_curr.loc[:, 'agnt_'].map(dict_agnt_prev_auto_nonRenew) 
    
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
    
#     # POL_LIFE
#     ten_feat = (pd.concat([dframe_curr['New-Renew Ind'], 
#                         dframe_curr['Effective YearMonth'].apply(str), 
#                         dframe_curr['Line Tenure'].apply(str)], axis=1)).apply(lambda x: ' '.join(x), axis=1)
    
#     ten = pd.DataFrame(ten_feat.apply(lambda x: NB_adj_tenure(x, period))).rename(columns={0:'tenure'})
    
#     df_ = pd.concat([dframe_curr, ten], axis=1) 
    
#     pol_feat = pd.concat([df_['New-Renew Ind'], 
#                         df_['Effective YearMonth'].apply(str), df_['tenure']],
#                         axis=1).apply(lambda x: ' '.join(x), axis=1)
    
#     pol_feat = pol_feat.apply(lambda x: delta(x, period))
    
#     dframe_curr['pol_life'] = pd.DataFrame(pol_feat).rename(columns={0:'pol_life'})
    
    '''Functions are located above. Splits up by line'''
    
#     dframe_curr = line_tot_prem(dframe_curr, q_num_curr, dframe_prev, q_num_prev)
        
#     dframe_curr = line_num_acc(dframe_curr, q_num_curr, dframe_prev, q_num_prev)
    
    dframe_curr = line_num_nonrenew_nb(dframe_curr, q_num_curr, dframe_prev, q_num_prev)


            
    # CURRENT QUARTER CLIENT TOTAL PREM 
    curr_clt = pd.pivot_table(dframe_curr, values='prem_', index='pol_num', 
                                   columns = 'quarter_', aggfunc='sum').fillna(0)
    dict_clt = pd.Series(curr_clt[q_num_curr], index=curr_clt.index).to_dict()
    dframe_curr.loc[:, 'curr_clt_tot_prem'] = dframe_curr.loc[:, 'pol_num'].map(dict_clt)
       
    
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

def equal_feat(dframe):
    
    
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
    
    # POLICY NUMBER ANONYMIZE
    
    fake = Faker()
    pol_names  = defaultdict(fake.md5) 
    dframe['pol_num'] = dframe['pol_num'].apply(lambda x: pol_names[x])
    
    
    # TERRITORY
    # with the state, subdivided territory < 30 
    # REALLY CAN'T USE THIS. THE VALUES ARE TOO HIGH AND NOT ASSIGNED TO A PARTICULAR COUNTY
    

    return(dframe)
    
    
    



#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

















       