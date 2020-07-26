# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 20:19:04 2019
Convert Dukascopy to Ctrader history data

@author: Paolo
"""


# load the dataset

import pandas as pd
import seaborn as sns


#carica dati dal file in df
path_load='C:\\Users\\Paolo\\Documents\\Python_EURUSD_bar//EURGBP.csv'
path_save='C:\\Users\\Paolo\\Documents\\Python_EURUSD_bar//EURGBP_CTRADER.csv'
df = pd.read_csv(path_load,sep=",",decimal=",", engine='python')
#elimina le rows del finesettimana dove il volume è zero
df = df[df.Volume != 0]
#rinomna la prima colonna come Date
df = df.rename(columns={'Gmt time': 'Date'})
#duplica la colonna datetime e la inserisce nella posizione 2
time=df['Date']
df.insert(loc=1, column='Time', value=time)

#df['Date'] = pd.to_datetime(df['Date'],dayfirst=True).dt.date
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True).dt.strftime('%Y.%m.%d')
df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%H:%M')

df['Volume']=1
#mantiene solo 5 digiti di decimale
df.round(5) 

#seleziona indice combinato data e time
#df2.set_index('Date','Time')

df.to_csv(path_save,index=False,header=False,float_format='%.5f')

#mostra graficamente il risultato dei close ordinati di tutta la serie
sns.set(rc={'figure.figsize':(11, 4)})
x=df["Close"]
x.plot(linewidth=0.7);




