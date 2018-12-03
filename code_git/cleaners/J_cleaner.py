import pandas as pd
import numpy as np
from datetime import datetime



#---------------------------------------------------------------------   
def j_factors(col, dframe, tol):
        n =  len(dframe[col].value_counts())
        print('Number of factors: ', n, '\n')
        
        
        while True:
            try:
                if n >= tol:
                    ques = ' '
                    while not (ques.startswith('y') or ques.startswith('n')):

                        ques = input('The factors exceed your tolerance, do you wish to see a truncated version?').lower()
                    if ques.startswith('y'):
                        print(dframe[col].value_counts().head(10))
                        break
                    else:
                        print('cool')
                        break
                else:
                    print(dframe[col].value_counts())
                    break

            except ValueError:
                print('yes or no')

                
                
#---------------------------------------------------------------------                
      
   
'''Region dictionary for mapping'''

region = {'AK': 'western', 'AZ': 'western', 'CO': 'western', 'ID': 'western', 'MT': 'western',
          'NM': 'western','NV': 'western', 'OR': 'western', 'UT': 'western', 'WA': 'western', 
          'WY': 'western', 
          
          'IL': 'central', 'IN': 'central', 'IA': 'central', 'KY': 'central', 
          'MI': 'central', 'MN': 'central', 'NE': 'central', 'ND': 'central', 'OH': 'central', 
          'SD': 'central', 'WI': 'central',
          
          'AL': 'southern', 'FL': 'southern', 'GA': 'southern', 
          'MS': 'southern', 'NC': 'southern', 'SC': 'southern', 'TN': 'southern',
          
          'CT': 'northeast', 'MA': 'northeast',  'ME': 'northeast',  'NH': 'northeast',  
          'NY': 'northeast',  'RI': 'northeast',  'VT': 'northeast', 
         
          'DC': 'mid atlantic', 'DE': 'mid atlantic', 'MD': 'mid atlantic', 'NJ': 'mid atlantic', 
          'PA': 'mid atlantic', 'VA': 'mid atlantic', 'WV': 'mid atlantic',
         
          'AR': 'south central', 'KS': 'south central', 'LA': 'south central', 'MO': 'south central', 
          'OK': 'south central', 'TX': 'south central'}
    
#---------------------------------------------------------------------  



def sub_df(dframe):
    
    '''returns a dataframe of only numerical values'''
    num_df = []
    
    for col in dframe.columns:
        x = dframe[col].iloc[0]
        if ( (isinstance(x, np.float64)) | (isinstance(x, np.int64)) ):
            num_df.append(col)
    
    number_df = dframe[num_df]
    return number_df

#---------------------------------------------------------------------  


import time
from IPython.core.magics.execution import _format_time
from IPython.display import display as d
from IPython.display import Audio
from IPython.core.display import HTML
import numpy as np
import logging as log

def alert():
    """ makes sound on client using javascript (works with remote server) """      
    framerate = 44100
    duration=.05
    freq=300
    t = np.linspace(0,duration,framerate*duration)
    data = np.sin(2*np.pi*freq*t)
    d(Audio(data,rate=framerate, autoplay=True))
#     hide_audio()
    
