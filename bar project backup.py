# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:01:32 2020

@author: Paolo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:01:27 2020

@author: Paolo

estrae massimi e minimi significativi
visualizza la distribuzione oraria e del giorno del mese e di fasce di prezzo per individuare
se esiste una concentrazione di swwing in certi orari o giorni e per visualizzare se esistono
fasce di prezzo su cui i prezzi tendono a girare, questo sarebbe gia una specie di market profile o meglio
una specie di individuazione delle resistenze, tuttavia l obiettivo e vedere se questi teorici supporti resistenze
sono ripetitivi nel tempo

successivamente si passa ad analizzare il tipo di candele presenti sugli swing bull and bear, in modo da paragonarli
e formare le caratteristiche di due classi da contrapporre all andamento normale

dai tratti salienti delle candele negli swing si spera inoltre di individuare dei pattern
"""
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
#from datetime import timedelta, date

#carica dati dal file in df
df = pd.read_csv('C:\\Users\\Paolo\\Documents\\Python_EURUSD_bar//eurusd1h.csv',sep=";",decimal=",", engine='python')
#trasforma la colonna Date in Datetime con il suo formato
df['Date'] =  pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S')
#mostra tipo dati delle colonne di df
df.dtypes
df=df.sort_values(by=['Date'])
#avendo ordinato per data tuttavia l indice generale risulta alterato, percui viene resettato con resetindex
df=df.reset_index(drop=True)

time=df['Date']
df.insert(loc=1, column='Time', value=time)

#df['Date'] = pd.to_datetime(df['Date'],dayfirst=True).dt.date
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True).dt.strftime('%Y.%m.%d')
df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%H:%M')

#creo un dataframe provvisorio piccolino
df2=df[0:1000]

#mostra graficamente il risultato dei close ordinati di tutta la serie
# sns.set(rc={'figure.figsize':(11, 4)})
# x=df2["Close"]
# x2=df_H["Close"]
# x.plot(linewidth=0.7);
# x2.plot(linewidth=0.7);

df_H=pd.DataFrame(columns=["Date","Time","Open","High","Low","Close","Volume"])

#dimension e il nuero di barre avanti e dietro
dimension=48  
index=100

df_pastH = df2[index-dimension:index]
max_pastH=df_pastH['High'].max()
df_nextH = df2[index+1:index+dimension+1]
max_nextH=df_nextH['High'].max()
###

df_pastH = df2[index-dimension:index]
df_pastH = df_pastH.sort_values(by=['High'],ascending=False)
df_pastH = df_pastH.reset_index(drop=True)
df_pastH = df_pastH[0:1]
H_past=df_pastH.iloc[0,'High']

#calcola il massimo next su n barre
df_nextH = df2[index+1:index+dimension+1]
df_nextH = df_nextH.sort_values(by=['High'],ascending=False)
df_nextH = df_nextH.reset_index(drop=True)
df_nextH = df_nextH[0:1]
H_next=df_nextH.at[0,'High']

current_H=df2.at[index,'High']

df_H=df_H.append(df2.loc[index])

dimension=20  
#crea dataframe in bianco per i massimi
df_H=pd.DataFrame(columns=["Date","Time","Open","High","Low","Close","Volume"])

for index in range(0,1000):
    #esegui solo de index e superiore a dimension
    if index >= dimension:
        #calcola il massimo past su n barre
        #estrapola dataframe di n dati past    
        df_pastH = df2[index-dimension:index]
        max_pastH=df_pastH['High'].max()
        df_nextH = df2[index+1:index+dimension+1]
        max_nextH=df_nextH['High'].max()

        
        current_H=df2.at[index,'High']
        
        if current_H > max_pastH and current_H > max_nextH:
        
            df_H=df_H.append(df2.loc[index])
    
    #se il massimo attuale e maggiore di entrambi ho uno swing e appendo a dataframe dei massimi
    #informazioni sono data ora O H L C
    
    #calcola il minimo past su n barre
    
    #calcola il minimo next su n barre
    
    #se il minimo attuale e minore di entrambi ho uno swing e appendo a dataframe dei minimi
    #informazioni sono data ora OHLC
    
