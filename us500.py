# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 20:19:04 2019
carica dati row e prepara per una rete lstm 
finestre overlapping di uno in uno, dati normalizzati per ogni timestamp

@author: Paolo
"""


# load the dataset

import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
#from datetime import timedelta, date
#import datetime
import numpy as np
from numpy import array
import matplotlib.pyplot as plt



#import time
# Use seaborn style defaults and set the default figure size


#carica dati dal file in df
df = pd.read_csv('C:\\Users\\Paolo\\Documents\\Python_EURUSD_bar//eurusd1h.csv',sep=";",decimal=",", engine='python')
#trasforma la colonna Date in Datetime con il suo formato
df['Date'] =  pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S')
#mostra tipo dati delle colonne di df
df.dtypes
#renti Date l'indice della serie
#df2=df.set_index("Date")
#elimina tutte le colonne lascia solo Close
#df3=df.drop(columns=["Open","High","Low"])
#ordina df3 secondo il suo indice ovvero Date, visto che i dati vengono al contrario dall ultimo al primo
df=df.sort_values(by=['Date'])
#avendo ordinato per data tuttavia l indice generale risulta alterato, percui viene resettato con resetindex
df=df.reset_index(drop=True)
#mostra graficamente il risultato dei close ordinati di tutta la serie
sns.set(rc={'figure.figsize':(11, 4)})
x=df["Close"]
x.plot(linewidth=0.7);



