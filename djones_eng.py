# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:39:18 2020

@author: Paolo
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:15:30 2020

EURUSD verifica di Ctrader per dow jones

@author: Paolo
"""

import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
#from datetime import timedelta, date
import numpy as np


#carica dati dal file in df
df = pd.read_csv('C:\\Users\\Paolo\\Documents\\Python_EURUSD_bar//djx1h.csv',sep=";",decimal=",", engine='python')
#trasforma la colonna Date in Datetime con il suo formato
df['Date'] =  pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S')
#mostra tipo dati delle colonne di df
df.dtypes
df=df.sort_values(by=['Date'])
#avendo ordinato per data tuttavia l indice generale risulta alterato, percui viene resettato con resetindex
df=df.reset_index(drop=True)



#rows per calcolo max e min
rows_exp=42900
#8 anni 50 mila barre
#creo un dataframe provvisorio piccolino
df2=df[0:rows_exp]
df2=df2.drop("Volume",axis=1)

#raccogli i dati della barra passata, cosi quando estrapolo min e massimi ho i dati per vedere correlazioni e pattern a 3 barre
opens = df2["Open"].shift(1)
df2.insert( loc=1,column='Open_p', value=opens)

highs = df2["High"].shift(1)
df2.insert( loc=2,column='High_p', value=highs)

lows = df2["Low"].shift(1)
df2.insert( loc=3,column='Low_p', value=lows)

closes = df2["Close"].shift(1)
df2.insert( loc=4,column='Close_p', value=closes)

df2=df2[1:rows_exp]
df2=df2.reset_index(drop=True)


#ENGULFING OK ed Engulfing fail creo i due dataset
df_eng_ok=pd.DataFrame(columns=["Date","Open_p","High_p","Low_p","Close_p","Open","High","Low","Close"])
df_eng_fail=pd.DataFrame(columns=["Date","Open_p","High_p","Low_p","Close_p","Open","High","Low","Close"])

### REQUISITI ENGULFING
## room 6 bars esclusa quella prima della precedente
## past negative
## current positive
## ritorno current 01 02
## close current - open past < .15

## REQUISITI OK
## TP 1,7 STOP LOSS
## STOP LOSS CHIUSURA CORRENTE MENO MINIMO CORRENTE E PRECEDENTE

room=6
rit_m=0.1
rit_M=0.2
eng_level = 0.15
profit_factor = 1.7
a_stop = 0
a_profit = 0

for index in range(0,rows_exp+1):
    #esegui solo se index e uguale alla room, lo shift e le due barre del pattern, e finisci 3 giorni prima per avere margine
    if index >= room + 2 and index <= rows_exp-72:
        first_negative = df2.at[index , "Close_p"] < df2.at[index , "Open_p"]
        second_positive = df2.at[index , "Close"] > df2.at[index , "Open"]
        
        ritorno_last_perc = (df2.at[index , "Close"] - df2.at[index , "Open"]) / df2.at[index , "Close"] * 100
        ritorno_ok = ritorno_last_perc > rit_m and ritorno_last_perc < rit_M
        
        eng_diff_perc = (df2.at[index , "Close"] - df2.at[index , "Open_p"]) / df2.at[index , "Close"] * 100
        eng_ok = eng_diff_perc < eng_level and eng_diff_perc > 0
        
        df_room = df2[index-3-room : index-2]
        df_room_min = df_room['Low'].min()
        df_pattern = df2[index-1 : index+1]
        df_pattern_min = df_pattern['Low'].min()
        room_ok = df_pattern_min < df_room_min
        
        
        if first_negative and second_positive and ritorno_ok and eng_ok and room_ok :
        
            stop_loss = df_pattern_min
            current_price = df2.at[index , 'Close']
            range_pattern = current_price - stop_loss
            take_profit = current_price + (range_pattern * profit_factor)
            
            counter=1
            long=1
            
            while long==1:  
                high_l = df2.at[index + counter , 'High']
                low_l = df2.at[index + counter , 'Low']
                
                if low_l < stop_loss:
                    a_stop=a_stop+1
                    df_eng_fail=df_eng_fail.append(df2.loc[index])
                    long=0
                    break
                if high_l >= take_profit:
                    a_profit=a_profit+1
                    df_eng_ok=df_eng_ok.append(df2.loc[index])
                    long=0
                    break
                counter = counter + 1
                
   
    

            
