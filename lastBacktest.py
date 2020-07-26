# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:49:20 2020

@author: Paolo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:03:17 2020

@author: Paolo VALIDAZIONE DATI SWING SVM
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

df_validazione=df[rows_exp:len(df)] # df di validazione con i dati residui in questo caso da 35 mila in poi

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

# dimension_p=24

# dimension_n=48

room_size = 5
outcome_size = 50
a_stop=0
a_profit=0
a_none=0
profit_factor = 1.7

df_L=pd.DataFrame(columns=["Date","Open","High","Low","Close","Body","Shadow","Open_n","High_n","Low_n","Close_n","Body_n","Shadow_n","Open_p","High_p","Low_p","Close_p","Body_p","Shadow_p"])
df_Lfail=pd.DataFrame(columns=["Date","Open","High","Low","Close","Body","Shadow","Open_n","High_n","Low_n","Close_n","Body_n","Shadow_n","Open_p","High_p","Low_p","Close_p","Body_p","Shadow_p"])


for index in range(0,len(df2_B)):
    #esegui solo de index e superiore a dimension
    if index >= room_size+3 and index<=len(df2_B) - outcome_size -1:
        #ROOM
        df_pattern = df2_B[index-1 : index+2]
        max_pattern = df_pattern['High'].max()
        min_pattern = df_pattern['Low'].min()
        
        df_past = df2_B[index-1-room_size : index-1]
        min_past = df_past['Low'].min()
        
        room_ok = min_past > min_pattern
        
        #FIRST NEGATIVE AND SECOND POSITIVE
        first_neg = df2_B.at[index , 'Body_p'] < 0
        second_pos = df2_B.at[index , 'Body'] > 0
        
        #RITORNO OK 01 02
        ritorno_ok = df2_B.at[index , 'Body'] > 0.1 and df2_B.at[index , 'Body'] < 0.2
        
        #ENGULFING
        eng_ok = df2_B.at[index , 'Close'] > df2_B.at[index , 'Open_p']
        
        #THIRD MIN OK
        third_min_ok = df2_B.at[index , 'Low_n'] > min_pattern
        
        if room_ok and first_neg and second_pos and ritorno_ok and eng_ok and third_min_ok:
        
            stop_loss = min_pattern
            range_pattern = max_pattern - min_pattern
            take_profit = df2_B.at[index , 'Close_n'] + (range_pattern * profit_factor)
            # VERIFICA SE NELLE N BARRE SEGUENTI VIENE RAGGIUNTO IL TP O SI VA IN STOP O NIENTE
            df_outcome = df2_B[index + 2 : index+2+outcome_size]
            df_outcome = df_outcome.reset_index()
            for counter in range(0 , len(df_outcome)):
                if df_outcome.at[counter , 'Low'] < stop_loss:
                    a_stop=a_stop+1
                    df_Lfail=df_Lfail.append(df2_B.loc[index])
                    break
                if df_outcome.at[counter , 'High'] >= take_profit:
                    a_profit=a_profit+1
                    df_L=df_L.append(df2_B.loc[index])
                    break
                if counter == len(df_outcome)-1:
                    a_none = a_none + 1
                    df_Lfail=df_Lfail.append(df2_B.loc[index])
                    break
        
        
        
        
        
      
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


# from sklearn.neighbors import KNeighborsClassifier
# # #KNN
# modelSVC = KNeighborsClassifier(n_neighbors=2)
# # Train the model using the training sets


# from sklearn.tree import DecisionTreeRegressor 

# modelSVC= DecisionTreeRegressor(random_state = 0)

# # #SVM

from sklearn.svm import SVC

modelSVC=SVC(kernel="linear",C=10,class_weight={0:1, 1:2})

modelSVC.fit(X_train,y_train)

y_pred= modelSVC.predict(X_test)


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X_train, y_train)
print(clf.feature_importances_)

y_pred= clf.predict(X_test)

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



long_status=0
operations_in_loss=0
operations_in_profit=0

runup_over200=0


for index in range(0,len(df_bt)-24):
    if df_bt.at[index,'Category']==1 :
        #stop loss minimo delle barre -1 0 ed 1
        df_pattern = df_bt[index-1:index+2]
        stop_loss=df_pattern['Low'].min()     
        #take profit e 2 volte il range dal prezzo di ingresso al minimo di stop sommato al prezzo di ingresso
        
        #dataframe delle dimension n barre e calcolo del minimo, la predizione si fonda sul fatto che non verra rotto tale min
        df_nextBars=df_bt[index+2:index+dimension_n+1]
        min_nextBars=df_nextBars['Low'].min()
        
        
        if stop_loss < min_nextBars:
            operations_in_profit=operations_in_profit+1
                        
        elif stop_loss >= min_nextBars:
            operations_in_loss=operations_in_loss-1    
                


      
print('operazioni in Profit :'+ str(round(operations_in_profit)))     

print('operazioni in Loss :'+ str(round(operations_in_loss)))




tp_factor=2
long_status=0
operations_in_loss=0
operations_in_profit=0
equity=0
equity_curve=np.array(1)

somma_rangepatternLoss=0



for index in range(7,len(df_bt)-24):
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
                #print("LOSS - entry:"+str(round(entry_price,5))+" stop loss "+str(round(stop_loss,5))+" take profit "+str(round(take_profit,5))+" range pattern: "+str(round(range_pattern,5)))
                #print("---------------------------------------------------------------------------------------------")
                long_status=0
                    
            #se prendo il take profit aumento il contatore dei tp presi ed interrompo ciclo
            if df_bt.at[start_bar,'High']>=take_profit:
                somma_rangepatternLoss=somma_rangepatternLoss+range_pattern
                operations_in_profit=operations_in_profit+1
                equity=equity+1*tp_factor
                equity_curve=np.append(equity_curve,equity)
                #print("WIN - entry:"+str(round(entry_price,5))+" stop loss "+str(round(stop_loss,5))+" take profit "+str(round(take_profit,5))+" range pattern: "+str(round(range_pattern,5)))
                #print("---------------------------------------------------------------------------------------------")
                long_status=0
                    
            start_bar=start_bar+1

media_rangepatternLoss=somma_rangepatternLoss/operations_in_loss
       
print('operazioni in Profit :'+ str(round(operations_in_profit))+" guadagno punti "+str(operations_in_profit*0.00250))     

print('operazioni in Loss :'+ str(round(operations_in_loss))+ " perdita $ "+ str(operations_in_loss*media_rangepatternLoss))   
