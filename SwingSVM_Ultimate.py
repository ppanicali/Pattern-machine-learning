# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:57:06 2020

EURUSD decision tree percent norm
su tutto il campione e backtest di due anni e mezzo (0.7train)
score:0.79
              precision    recall  f1-score   support

           0       0.95      0.80      0.87      3533
           1       0.32      0.68      0.43       484

   micro avg       0.79      0.79      0.79      4017
   macro avg       0.63      0.74      0.65      4017
weighted avg       0.87      0.79      0.82      4017

[[2834  699]
 [ 157  327]]
operazioni in Profit :216 guadagno $ 497
operazioni in Loss :-260 perdita $ -260

curva di equity costante
-----------------------------------------------------------------------------
0-25000 0.8 di train e 0.2 test
score:0.82
              precision    recall  f1-score   support

           0       0.94      0.85      0.89       987
           1       0.38      0.64      0.48       142

   micro avg       0.82      0.82      0.82      1129
   macro avg       0.66      0.74      0.68      1129
weighted avg       0.87      0.82      0.84      1129

[[837 150]
 [ 51  91]]
operazioni in Profit :55 guadagno $ 126
operazioni in Loss :-57 perdita $ -57
------------------------------------------------------------------------------

10000-35000
score:0.72
              precision    recall  f1-score   support

           0       0.97      0.70      0.82       984
           1       0.28      0.84      0.42       138

   micro avg       0.72      0.72      0.72      1122
   macro avg       0.63      0.77      0.62      1122
weighted avg       0.88      0.72      0.77      1122

[[692 292]
 [ 22 116]]
operazioni in Profit :103 guadagno $ 237
operazioni in Loss :-110 perdita $ -110

-------------------------------------------------------------------------------
20000-45000
score:0.80
score:0.78
              precision    recall  f1-score   support

           0       0.95      0.79      0.86       980
           1       0.36      0.72      0.48       160

   micro avg       0.78      0.78      0.78      1140
   macro avg       0.65      0.76      0.67      1140
weighted avg       0.86      0.78      0.80      1140

[[770 210]
 [ 44 116]]
operazioni in Profit :162 guadagno $ 373
operazioni in Loss :-156 perdita $ -156
------------------------------------------------------------------------------
30000-55000
score:0.77
              precision    recall  f1-score   support

           0       0.94      0.79      0.86      1006
           1       0.30      0.63      0.41       145

   micro avg       0.77      0.77      0.77      1151
   macro avg       0.62      0.71      0.63      1151
weighted avg       0.86      0.77      0.80      1151

[[795 211]
 [ 53  92]]

operazioni in Profit :58 guadagno $ 133
operazioni in Loss :-91 perdita $ -91

------------------------------------------------------------------------------
23000 +35000
score:0.80
              precision    recall  f1-score   support

           0       0.94      0.83      0.88      1420
           1       0.32      0.60      0.42       193

   micro avg       0.80      0.80      0.80      1613
   macro avg       0.63      0.71      0.65      1613
weighted avg       0.86      0.80      0.82      1613

[[1176  244]
 [  77  116]]
operazioni in Profit :70 guadagno $ 161
operazioni in Loss :-102 perdita $ -102
----------------------------------------------------------

0-58600   0,6 train e 0,4 test

mag 2016 feb 2020 3 anni e mezzo  660 operazioni, hit rate 0.5 profit factor 1,99

operazioni in Profit :329 guadagno $ 658
operazioni in Loss :-330 perdita $ -330

score:0.79
              precision    recall  f1-score   support

           0       0.95      0.80      0.87      4714
           1       0.32      0.69      0.44       642

   micro avg       0.79      0.79      0.79      5356
   macro avg       0.63      0.75      0.65      5356
weighted avg       0.87      0.79      0.82      5356

[[3764  950]
 [ 198  444]]

--------------------------------------------------------------
marzo 14 maggio 16 campione 0-35000 con train 0.6 percento
operazioni in Profit :260 guadagno $ 520
operazioni in Loss :-250 perdita $ -250 curva perfetta
26 mesi 510 operazioni ovvero 20 al mese ed in piena caduta libera
--------------------------------------------------------------------------

@author: Paolo
"""

##CARICA I DATI DELLA TIME SERIE
import pandas as pd
import numpy as np

#carica dati dal file in df
df = pd.read_csv('C:\\Users\\Paolo\\Documents\\Python_EURUSD_bar//eurusd1h.csv',sep=";",decimal=",", engine='python')
#trasforma la colonna Date in Datetime con il suo formato
df['Date'] =  pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S')
#mostra tipo dati delle colonne di df
df.dtypes
df=df.sort_values(by=['Date'])
#avendo ordinato per data tuttavia l indice generale risulta alterato, percui viene resettato con resetindex
df=df.reset_index(drop=True)


#rows per calcolo max e min
start_bar=0
rows_exp=start_bar+35000
#8 anni 50 mila barre
#creo un dataframecon il numero di barre indicato dalla variabile rows exp
df2=df[start_bar:rows_exp]

df_future=df[rows_exp:rows_exp+6000] # un anno in avanti per un test alla fine

df2=df2.drop(['Volume'],axis=1)
df2=df2.reset_index(drop=True)
# indico quande barre prima e dopo definiscono il minimo rilevante

#COPIA PER FARE BACKTESTING ALLA FINE
df_Backtest=df2.copy()


## CREA LE COLONNE OHLC NEXT E PAST PER AVERE SU OGNI ROW IL PATTERN A TRE BARRE COMPLETO
#raccogli i dati della barra seguente, cosi quando estrapolo min e massimi ho i dati per vedere correlazioni e pattern a 3 barre
df2['Open_n']=df2["Open"].shift(-1)
df2['High_n']=df2["High"].shift(-1)
df2['Low_n']=df2["Low"].shift(-1)
df2['Close_n']=df2["Close"].shift(-1)
#raccogli i dati della barra passata, cosi quando estrapolo min e massimi ho i dati per vedere correlazioni e pattern a 3 barre
df2['Open_p']=df2["Open"].shift(1)
df2['High_p']=df2["High"].shift(1)
df2['Low_p']=df2["Low"].shift(1)
df2['Close_p']=df2["Close"].shift(1)

####################################

#### DATAFRAME dei minimi a cui viene aggiunta la dimensione percentuale della barra ohlc,_p,_n
df2_B=df2.copy()
#aggiungi colonna di ritorni e fai la percentuale rispetto alla chiusura attuale
Bodies=df2_B['Close']-df2_B['Open']
df2_B.insert( loc=5,column='Body', value=Bodies)
df2_B['Body'] = df2_B['Body'] / df2_B['Close']*100  

#aggiungi colonna di ritorni dei next e fai la percentuale rispetto alla chiusura attuale
Bodies=df2_B['Close_n']-df2_B['Open_n']
df2_B.insert( loc=10,column='Body_n', value=Bodies)
df2_B['Body_n'] = df2_B['Body_n'] / df2_B['Close']*100 

#aggiungi colonna di ritorni dei past e fai la percentuale rispetto alla chiusura attuale
Bodies=df2_B['Close_p']-df2_B['Open_p']
df2_B.insert( loc=15,column='Body_p', value=Bodies)
df2_B['Body_p'] = df2_B['Body_p'] / df2_B['Close']*100 

#################### SHADOW DEI MINIMI
############################################################################################
# SHADOW CURRENT
df2_B.insert( loc=6,column='Shadow',value=0)
for index in range(0,len(df2_B.index)):
    if df2_B.at[index,'Close'] >= df2_B.at[index,'Open']:
        wick_size=df2_B.at[index,'Open'] - df2_B.at[index,'Low'] 
    elif df2_B.at[index,'Open'] > df2_B.at[index,'Close']:
        wick_size=df2_B.at[index,'Close'] - df2_B.at[index,'Low']
    df2_B.loc[index,'Shadow']=wick_size
  
df2_B['Shadow'] = df2_B['Shadow'] / df2_B['Close']*100  

## SHADOW PAST
df2_B.insert( loc=12,column='Shadow_n',value=0)
for index in range(0,len(df2_B.index)):
    if df2_B.at[index,'Close_n'] >= df2_B.at[index,'Open_n']:
        wick_size=df2_B.at[index,'Open_n'] - df2_B.at[index,'Low_n'] 
    elif df2_B.at[index,'Open_n'] > df2_B.at[index,'Close_n']:
        wick_size=df2_B.at[index,'Close_n'] - df2_B.at[index,'Low_n']
    df2_B.loc[index,'Shadow_n']=wick_size
  
df2_B['Shadow_n'] = df2_B['Shadow_n'] / df2_B['Close_n']*100 

## SHADOW PAST
df2_B.insert( loc=18,column='Shadow_p',value=0)
for index in range(0,len(df2_B.index)):
    if df2_B.at[index,'Close_p'] >= df2_B.at[index,'Open_p']:
        wick_size=df2_B.at[index,'Open_p'] - df2_B.at[index,'Low_p'] 
    elif df2_B.at[index,'Open_p'] > df2_B.at[index,'Close_p']:
        wick_size=df2_B.at[index,'Close_p'] - df2_B.at[index,'Low_p']
    df2_B.loc[index,'Shadow_p']=wick_size
  
df2_B['Shadow_p'] = df2_B['Shadow_p'] / df2_B['Close_p']*100  

####################################

## CREA UN DATAFRAME DI MASSIMI E MINIMI E DI MASSIMI E MINIMI FALLITI, CHE SONO QUELLI CHE HANNO SOLO LA ROOM A SINISTRA
## DF_H E DF_HFAIL  E DI CONVERSO DF_L E DF_LFAIL

dimension_p=48

dimension_n=48

df_L=pd.DataFrame(columns=["Date","Open","High","Low","Close","Body","Shadow","Open_n","High_n","Low_n","Close_n","Body_n","Shadow_n","Open_p","High_p","Low_p","Close_p","Body_p","Shadow_p"])
df_Lfail=pd.DataFrame(columns=["Date","Open","High","Low","Close","Body","Shadow","Open_n","High_n","Low_n","Close_n","Body_n","Shadow_n","Open_p","High_p","Low_p","Close_p","Body_p","Shadow_p"])


for index in range(0,len(df2_B)):
    #esegui solo de index e superiore a dimension
    if index >= dimension_p and index<=rows_exp-dimension_n:
        ## MINIMI VERI
        #dataframe con i dati precedenti al valore attuale 
        df_pastL = df2_B[index-dimension_p:index]
        #estrapolo il minimo da i dimension dati prima
        min_pastL=df_pastL['Low'].min()
        #dataframe con i dati successivi al valore attuale
        df_nextL = df2_B[index+1:index+dimension_n+1]
        #estrapolo il minimo da i dimension dati successivi
        min_nextL=df_nextL['Low'].min()
        #se il min attuale e inferiore ai dimension prima ed i dimension dopo lo appendo al dataframe dei minimi
        current_L=df2_B.at[index,'Low']       
        if current_L < min_pastL and current_L < min_nextL:      
            df_L=df_L.append(df2_B.loc[index])
        ## MINIMI FALLITI    
        if current_L < min_pastL and current_L > min_nextL:
            df_Lfail=df_Lfail.append(df2_B.loc[index])    
   
    
      
###################################################################

df_L['Category']=1
df_Lfail['Category']=0


####################################################################################

## DATI PRONTI PER MACHINE LEARNING CON BODY E SHADOW ESSENDO MINIMI
## DATAFRAME DF_l2 E DF_L2 FAIL

#################################################################################3


##creo copia dei database con i minimi ed i minii falliti con indice la data per usare con SVM
## creo anche una copia di tutta la timeseries con indice la data per visualizzare tutto graficamente
df_Lfail_DateIndex = df_Lfail.set_index('Date')
df_L_DateIndex = df_L.set_index('Date')
df2_B_DateIndex=df2_B.set_index('Date')
##########################################################################################################

## VISUALIZZA GRAFICAMENTE IL GRAFICO LINEA DELLE CHIUSURE TUTTE QUELLE DEL CAMPIONE ED I PUNTI DI MINIMO
## INDIVIDUATI E CONTENUTI NEL DATAFRAME DF_L2

CloseData=df2_B_DateIndex['Low']
CloseLine=CloseData.plot()

Lows=df_L_DateIndex['Low']-0.00100
LowsPlots=Lows.plot(style='k.')
###########################################################################################################3

## MERGE I DUE DATAFRAME PER CERCARE CATEGORIE A MINIMI E B MINIMI FALLITI, VA SHUFFLATO
## drop le colonne di o h l c e volume
frames = [df_L_DateIndex.copy(),df_Lfail_DateIndex.copy()]

df_collezione_min = pd.concat(frames)

#elimina tutte le colonne di valori tranne Low che mi serve alla fine per visualizzare graficamente i punti di inversione predetti lascio anche close next
#per poter fare backtesting
df_collezione_min=df_collezione_min.drop(["Open","High","Low","Close","Open_n","High_n","Low_n","Close_n","Open_p","High_p","Low_p","Close_p"], axis=1)

## PREPARA DATI PER LA PREDIZIONE TENENDOLI ORDINATI CRONOLOGICAMENTE

df_collezione_min=df_collezione_min.sort_values(by=['Date'])

## percentuale di train 80% 0.8 in questo caso
df_collezioneTrain=df_collezione_min[0:round(len(df_collezione_min)*0.6)]
df_collezioneTest=df_collezione_min[round(len(df_collezione_min)*0.6):len(df_collezione_min)]

y_train = df_collezioneTrain.Category
X_train = df_collezioneTrain.drop('Category',axis=1)

y_test = df_collezioneTest.Category
X_test = df_collezioneTest.drop('Category',axis=1)


# #SVM

from sklearn.svm import SVC

modelSVC=SVC(kernel="linear",C=10,class_weight={0:1, 1:8})

modelSVC.fit(X_train,y_train)

y_pred= modelSVC.predict(X_test)

########################################################################3

from sklearn.metrics import classification_report
##valore medio di accuracy##
print("score:{:.2f}".format(np.mean(y_pred == y_test)))
print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix

confusion=confusion_matrix(y_test, y_pred)
print(confusion)

##VISUALIZZAZIONE GRAFICA DELLE PREVISIONI
#copia il dataframe dove sono i dati x mandati a machine learn con l indice giusto
X_test_copy=X_test.copy()
#unisci la colonna delle predizioni cosi ho le predizioni di 1 allineate con indice corretto
X_test_copy['Category']=y_pred
X_Backtesting_signals=X_test_copy.drop(['Body','Body_n','Body_p','Shadow','Shadow_n','Shadow_p'],axis=1)
## solo i long
X_Backtesting_signals_long=X_Backtesting_signals.loc[X_Backtesting_signals['Category']==1] 
## aggiungi ohlc al dataframe dei segnali solo long predetti
df_BacktestD=df_Backtest.set_index('Date')
Long_pred=X_Backtesting_signals.combine_first(df_BacktestD)

##Long pred sono le barre long con ohlc con indice data
Long_pred=Long_pred.loc[Long_pred['Category']==1]

#CREAZIONE GRAFICO CON SERIE ORIGINALE, MINIMI CALCOLATI E MINIMI PREDETTI
CloseData=df2_B_DateIndex['Low']
CloseLine=CloseData.plot()

LowsPredetti=Long_pred['Close']
LowsPlots=LowsPredetti.plot(style='r.')

# Lows=df_L2D['Low']
# LowsPlots=Lows.plot(style='k.')


## prepara per il backtesting la serie df_bt
df_bt=X_Backtesting_signals.combine_first(df2_B_DateIndex)
df_bt=df_bt.reset_index()

tp_factor=2
long_status=0
operations_in_loss=0
operations_in_profit=0
equity=0
equity_curve=np.array(1)



for index in range(0,len(df_bt)-24):
    if df_bt.at[index,'Category']==1 :
        long_status=1
        entry_price=df_bt.at[index,'Close_n']
        #stop loss minimo delle barre -1 0 ed 1
        df_pattern = df_bt[index-1:index+2]
        stop_loss=df_pattern['Low'].min()
        
        #take profit e 2 volte il range dal prezzo di ingresso al minimo di stop sommato al prezzo di ingresso
        range_pattern=entry_price-stop_loss
        take_profit=entry_price+range_pattern*tp_factor
        
        #ciclo indefinito while dalla barra index+2 in poi per vedere se si prende lo stop o il tp
        start_bar=index+2
        while long_status==1:   
            #se prendo stop aumento il contatore degli stop presi ed interrompo ciclo
            if df_bt.at[start_bar,'Low']<=stop_loss:
                operations_in_loss=operations_in_loss-1
                equity=equity-1
                equity_curve=np.append(equity_curve,equity)
                print("LOSS - entry:"+str(round(entry_price,5))+" stop loss "+str(round(stop_loss,5))+" take profit "+str(round(take_profit,5))+" range pattern: "+str(round(range_pattern,5)))
                print("---------------------------------------------------------------------------------------------")
                long_status=0
                    
            #se prendo il take profit aumento il contatore dei tp presi ed interrompo ciclo
            if df_bt.at[start_bar,'High']>=take_profit:
                operations_in_profit=operations_in_profit+1
                equity=equity+1*tp_factor
                equity_curve=np.append(equity_curve,equity)
                print("WIN - entry:"+str(round(entry_price,5))+" stop loss "+str(round(stop_loss,5))+" take profit "+str(round(take_profit,5))+" range pattern: "+str(round(range_pattern,5)))
                print("---------------------------------------------------------------------------------------------")
                long_status=0
                    
            start_bar=start_bar+1

       
print('operazioni in Profit :'+ str(round(operations_in_profit))+" guadagno $ "+str(round(operations_in_profit*tp_factor)))     

print('operazioni in Loss :'+ str(round(operations_in_loss))+ " perdita $ "+ str(round(operations_in_loss)))   

df_equity=pd.DataFrame(columns=["EquityLine"])
df_equity['EquityLine']=equity_curve
EquityLine=df_equity['EquityLine']
CloseLine=EquityLine.plot()

from sklearn.externals import joblib
joblib.dump(modelSVC, 'modelEURUSD0-35kbars.pkl')
    
#DISTRIBUZIONE DEI BODI NELLE CANDELE DI MINIMO
bodyGraph=df_Lfail['Shadow_n'].plot.hist(title='body size',bins=50,alpha=0.5,color='blue',normed=True)   
#bodyGraph2=df_serie['Body'].plot.hist(title='body size',bins=300,alpha=0.2,color='red', normed=True) 

         
