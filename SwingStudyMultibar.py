# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:01:27 2020

@author: Paolo

Estrae anche i dati della barra successiva non solo della barra di minimo o massimo


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

time=df['Date']
df.insert(loc=1, column='Time', value=time)

#raccogli i dati della barra seguente, cosi quando estrapolo min e massimi ho i dati per vedere correlazioni e pattern a 3 barre
df['Open_n']=df["Open"].shift(-1)
df['High_n']=df["High"].shift(-1)
df['Low_n']=df["Low"].shift(-1)
df['Close_n']=df["Close"].shift(-1)

#raccogli i dati della barra passata, cosi quando estrapolo min e massimi ho i dati per vedere correlazioni e pattern a 3 barre
df['Open_p']=df["Open"].shift(1)
df['High_p']=df["High"].shift(1)
df['Low_p']=df["Low"].shift(1)
df['Close_p']=df["Close"].shift(1)


#df['Date'] = pd.to_datetime(df['Date'],dayfirst=True).dt.strftime('%Y.%m.%d')
#df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%H:%M')

#rows per calcolo max e min
rows_exp=20000
#8 anni 50 mila barre
#creo un dataframe provvisorio piccolino
df2=df[0:rows_exp]

dimension=12  
#crea dataframe in bianco per i massimi CONTEMPLO ORA ANCHE I DATI DELLA BARRA NEXT PER VALUTARE PATTERN DUE BARRE
df_H=pd.DataFrame(columns=["Date","Time","Open","High","Low","Close","Volume","Open_n","High_n","Low_n","Close_n","Open_p","High_p","Low_p","Close_p"])
df_L=pd.DataFrame(columns=["Date","Time","Open","High","Low","Close","Volume","Open_n","High_n","Low_n","Close_n","Open_p","High_p","Low_p","Close_p"])

#crea dataframe in bianco per i mancati minimi ed i mancati massimi da classificare rispetto a quelli che sono veramente max o min
df_Hfail=pd.DataFrame(columns=["Date","Time","Open","High","Low","Close","Volume","Open_n","High_n","Low_n","Close_n","Open_p","High_p","Low_p","Close_p"])
df_Lfail=pd.DataFrame(columns=["Date","Time","Open","High","Low","Close","Volume","Open_n","High_n","Low_n","Close_n","Open_p","High_p","Low_p","Close_p"])


for index in range(0,rows_exp+1):
    #esegui solo de index e superiore a dimension
    if index >= dimension and index<=rows_exp-dimension:
        
        ### MASSIMI VERI
        #calcola il massimo past su n barre
        #dataframe con i dati precedenti al valore attuale 
        df_pastH = df2[index-dimension:index]
        #estrapolo il massimo da i dimension dati prima
        max_pastH=df_pastH['High'].max()
        #dataframe con i dati successivi al valore attuale
        df_nextH = df2[index+1:index+dimension+1]
        #estrapolo il massimo da i dimension dati successivi
        max_nextH=df_nextH['High'].max()
        #se il massimo attuale e superiore ai dimension prima ed i dimension dopo lo appendo al dataframe dei massimi
        current_H=df2.at[index,'High']       
        if current_H > max_pastH and current_H > max_nextH:
        
            df_H=df_H.append(df2.loc[index])
            
        ### MASSIMI FALLITI         
        if current_H > max_pastH and current_H < max_nextH:
        
            df_Hfail=df_Hfail.append(df2.loc[index])    
            
        
        #dataframe con i dati precedenti al valore attuale 
        df_pastL = df2[index-dimension:index]
        #estrapolo il minimo da i dimension dati prima
        min_pastL=df_pastL['Low'].min()
        #dataframe con i dati successivi al valore attuale
        df_nextL = df2[index+1:index+dimension+1]
        #estrapolo il minimo da i dimension dati successivi
        min_nextL=df_nextL['Low'].min()
        #se il min attuale e inferiore ai dimension prima ed i dimension dopo lo appendo al dataframe dei minimi
        current_L=df2.at[index,'Low']       
        if current_L < min_pastL and current_L < min_nextL:
        
            df_L=df_L.append(df2.loc[index])
            
        if current_L < min_pastL and current_L > min_nextL:
        
            df_Lfail=df_Lfail.append(df2.loc[index])    
   
    
  
#MOSTRA RISULTATI DI MASSIMI E MINIMI TROVATI
sns.set(rc={'figure.figsize':(11, 4)})
x_H=df2["High"]
x_L=df2["Low"]

x2_H=df_H["High"]
x2_L=df_L["Low"]

x_H.plot(linewidth=0.7);
x2_H.plot(linewidth=0.7);   

x_L.plot(linewidth=0.7);
x2_L.plot(linewidth=0.7); 

#######################################################





##CALCOLA BODY, WICK E SHADOW DEL DATABASE DEI MASSIMI IN PERCENTUALE
# df_H2=df_H.copy()

#BODY 

# ##### DATAFRAME DEI MASSIMI BODY
# #aggiungi colonna di ritorni e fai la percentuale rispetto alla chiusura attuale
# Bodies=df_H2['Close']-df_H2['Open']
# df_H2.insert( loc=6,column='Body', value=Bodies)
# df_H2['Body'] = df_H2['Body'] / df_H2['Close']*100  

# #aggiungi colonna di ritorni dei next e fai la percentuale rispetto alla chiusura attuale
# Bodies=df_H2['Close_n']-df_H2['Open_n']
# df_H2.insert( loc=12,column='Body_n', value=Bodies)
# df_H2['Body_n'] = df_H2['Body_n'] / df_H2['Close']*100 

# #aggiungi colonna di ritorni dei past e fai la percentuale rispetto alla chiusura attuale
# Bodies=df_H2['Close_p']-df_H2['Open_p']
# df_H2.insert( loc=17,column='Body_p', value=Bodies)
# df_H2['Body_p'] = df_H2['Body_p'] / df_H2['Close']*100 



#### DATAFRAME DEI MINIMI BODY
df_L2=df_L.copy()
#aggiungi colonna di ritorni e fai la percentuale rispetto alla chiusura attuale
Bodies=df_L2['Close']-df_L2['Open']
df_L2.insert( loc=6,column='Body', value=Bodies)
df_L2['Body'] = df_L2['Body'] / df_L2['Close']*100  

#aggiungi colonna di ritorni dei next e fai la percentuale rispetto alla chiusura attuale
Bodies=df_L2['Close_n']-df_L2['Open_n']
df_L2.insert( loc=12,column='Body_n', value=Bodies)
df_L2['Body_n'] = df_L2['Body_n'] / df_L2['Close']*100 

#aggiungi colonna di ritorni dei past e fai la percentuale rispetto alla chiusura attuale
Bodies=df_L2['Close_p']-df_L2['Open_p']
df_L2.insert( loc=17,column='Body_p', value=Bodies)
df_L2['Body_p'] = df_L2['Body_p'] / df_L2['Close']*100 

df_L2['Category']=1

#### DATAFRAME DEI MINIMI FALLITI BODY
df_Lfail2=df_Lfail.copy()
#aggiungi colonna di ritorni e fai la percentuale rispetto alla chiusura attuale
Bodies=df_Lfail2['Close']-df_Lfail2['Open']
df_Lfail2.insert( loc=6,column='Body', value=Bodies)
df_Lfail2['Body'] = df_Lfail2['Body'] / df_Lfail2['Close']*100  

#aggiungi colonna di ritorni dei next e fai la percentuale rispetto alla chiusura attuale
Bodies=df_Lfail2['Close_n']-df_Lfail2['Open_n']
df_Lfail2.insert( loc=12,column='Body_n', value=Bodies)
df_Lfail2['Body_n'] = df_Lfail2['Body_n'] / df_Lfail2['Close']*100 

#aggiungi colonna di ritorni dei past e fai la percentuale rispetto alla chiusura attuale
Bodies=df_Lfail2['Close_p']-df_Lfail2['Open_p']
df_Lfail2.insert( loc=17,column='Body_p', value=Bodies)
df_Lfail2['Body_p'] = df_Lfail2['Body_p'] / df_Lfail2['Close']*100 
 

df_Lfail2['Category']=0

## VISUALIZZA GRAFICAMENTE IL GRAFICO LINEA DELLE CHIUSURE TUTTE QUELLE DEL CAMPIONE ED I PUNTI DI MINIMO
## INDIVIDUATI E CONTENUTI NEL DATAFRAME DF_L2
CloseData=df2['Low']
CloseLine=CloseData.plot()

Lows=df_L2['Low']
LowsPlots=Lows.plot(style='k.')


## MERGE I DUE DATAFRAME PER CERCARE CATEGORIE A MINIMI E B MINIMI FALLITI, VA SHUFFLATO
## drop le colonne di o h l c e volume
frames = [df_Lfail2.copy(),df_L2.copy()]

df_collezione_min = pd.concat(frames)

#elimina tutte le colonne di valori tranne Low che mi serve alla fine per visualizzare graficamente i punti di inversione predetti
df_collezione_min=df_collezione_min.drop(["Date","Time","Open","High","Close","Volume","Open_n","High_n","Low_n","Close_n","Open_p","High_p","Low_p","Close_p","Volume"], axis=1)


#ml non resetto altrimenti perdo l indice e alla fine non posso vedere i risultati
#df_collezione_min=df_collezione_min.reset_index(drop=True)

from sklearn.model_selection import train_test_split

y = df_collezione_min.Category
X = df_collezione_min.drop('Category',axis=1)

##effettua lo split dei dati

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5)

# from sklearn.neighbors import KNeighborsClassifier

# #KNN
# model = KNeighborsClassifier(n_neighbors=8)
# # Train the model using the training sets
# model.fit(X_train,y_train)

#SVC
from sklearn.svm import SVC

modelSVC=SVC(kernel="sigmoid",C=3000,random_state=10000,gamma='auto',class_weight={0: 2, 1: 1})
modelSVC.fit(X_train,y_train)

y_pred= modelSVC.predict(X_test)
print(y_pred)

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
#estrapola solo gli 1 che sono quelli da visualizzare scartando gli 0 
X_lowPredicted=X_test_copy.loc[X_test_copy['Category']==1] 
#a questo punto posso visualizzare x_low predicted sul grafico con i close di tutta la serie per vedere i predetti, confrontarli con quelli 
#effettivi e la timeseries anche

#CREAZIONE GRAFICO CON SERIE ORIGINALE, MINIMI CALCOLATI E MINIMI PREDETTI
CloseData=df2['Low']
CloseLine=CloseData.plot()

LowsPredetti=X_lowPredicted['Low']
LowsPlots=LowsPredetti.plot(style='k.')




# ###### DATAFRAME DI TUTTI I DATI

# df_serie=df2.copy()
# #aggiungi colonna di ritorni e fai la percentuale rispoetto chiusura corrente
# Bodies=df_serie['Close']-df_serie['Open']
# df_serie.insert( loc=6,column='Body', value=Bodies)
# df_serie['Body'] = df_serie['Body'] / df_serie['Close']*100

# #aggiungi colonna dei ritorni next e fai la percentuale rispetto alla chiusura corrente
# Bodies=df_serie['Close_n']-df_serie['Open_n']
# df_serie.insert( loc=12,column='Body_n', value=Bodies)
# df_serie['Body_n'] = df_serie['Body_n'] / df_serie['Close']*100

# #aggiungi colonna dei ritorni past e fai la percentuale rispetto alla chiusura corrente
# Bodies=df_serie['Close_p']-df_serie['Open_p']
# df_serie.insert( loc=17,column='Body_p', value=Bodies)
# df_serie['Body_p'] = df_serie['Body_p'] / df_serie['Close']*100




#### GRAFICO SCATTER BODY BS BODY NEXT
Scatter_Wick_body=df_L2.plot.scatter(x='Body',y='Body_n',alpha=0.5)

Scatter_Wick_body=df_Lfail2.plot.scatter(x='Body',y='Body_n',alpha=0.5)

Scatter_Wick_body=df_collezione_min.plot.scatter(x='Body',y='Body_n',alpha=0.5,c='Category')





























#aggiungi colonna di wick
df_H2.insert( loc=8,column='Wick',value=0)
df_H2=df_H2.reset_index(drop=True)

for index in range(0,len(df_H2.index)):
    if df_H2.at[index,'Close'] >= df_H2.at[index,'Open']:
        wick_size=df_H2.at[index,'High'] - df_H2.at[index,'Close'] 
    elif df_H2.at[index,'Open'] > df_H2.at[index,'Close']:
        wick_size=df_H2.at[index,'High'] - df_H2.at[index,'Open']
    df_H2.loc[index,'Wick']=wick_size    

#aggiungi colonna di shadow
df_H2.insert( loc=9,column='Shadow',value=0)
for index in range(0,len(df_H2.index)):
    if df_H2.at[index,'Close'] >= df_H2.at[index,'Open']:
        wick_size=df_H2.at[index,'Open'] - df_H2.at[index,'Low'] 
    elif df_H2.at[index,'Open'] > df_H2.at[index,'Close']:
        wick_size=df_H2.at[index,'Close'] - df_H2.at[index,'Low']
    df_H2.loc[index,'Shadow']=wick_size

 
df_H2['Wick'] = df_H2['Wick'] / df_H2['Close']*100 
df_H2['Shadow'] = df_H2['Shadow'] / df_H2['Close']*100     






#CALCOLA BODY, WICK E SHADOW DEL DATAFRAME DEI MINIMI

df_L2=df_L.copy()

#aggiungi colonna di ritorni
Bodies=df_L2['Close']-df_L2['Open']
df_L2.insert( loc=7,column='Body', value=Bodies)

#aggiungi colonna di wick
df_L2.insert( loc=8,column='Wick',value=0)
df_L2=df_L2.reset_index(drop=True)

for index in range(0,len(df_L2.index)):
    if df_L2.at[index,'Close'] >= df_L2.at[index,'Open']:
        wick_size=df_L2.at[index,'High'] - df_L2.at[index,'Close'] 
    elif df_L2.at[index,'Open'] > df_L2.at[index,'Close']:
        wick_size=df_L2.at[index,'High'] - df_L2.at[index,'Open']
    df_L2.loc[index,'Wick']=wick_size    

#aggiungi colonna di shadow
df_L2.insert( loc=9,column='Shadow',value=0)
for index in range(0,len(df_L2.index)):
    if df_L2.at[index,'Close'] >= df_L2.at[index,'Open']:
        wick_size=df_L2.at[index,'Open'] - df_L2.at[index,'Low'] 
    elif df_L2.at[index,'Open'] > df_L2.at[index,'Close']:
        wick_size=df_L2.at[index,'Close'] - df_L2.at[index,'Low']
    df_L2.loc[index,'Shadow']=wick_size

df_L2['Body'] = df_L2['Body'] / df_L2['Close']*100   
df_L2['Wick'] = df_L2['Wick'] / df_L2['Close']*100 
df_L2['Shadow'] = df_L2['Shadow'] / df_L2['Close']*100     
    





#CALCOLA BODY WICK E SHADOW DEL DATAFRAME CON TUTTO IL CAMPIONE PER COMPARARE CON QUELLI DI MAX E MINIMI

#aggiungi colonna di ritorni
df_serie=df2.copy()
Bodies=df_serie['Close']-df_serie['Open']
df_serie.insert( loc=7,column='Body', value=Bodies)

#aggiungi colonna di wick
df_serie.insert( loc=8,column='Wick',value=0)
df_serie=df_serie.reset_index(drop=True)

for index in range(0,len(df_serie.index)):
    if df_serie.at[index,'Close'] >= df_serie.at[index,'Open']:
        wick_size=df_serie.at[index,'High'] - df_serie.at[index,'Close'] 
    elif df_serie.at[index,'Open'] > df_serie.at[index,'Close']:
        wick_size=df_serie.at[index,'High'] - df_serie.at[index,'Open']
    df_serie.loc[index,'Wick']=wick_size    

#aggiungi colonna di shadow
df_serie.insert( loc=9,column='Shadow',value=0)
for index in range(0,len(df_serie.index)):
    if df_serie.at[index,'Close'] >= df_serie.at[index,'Open']:
        wick_size=df_serie.at[index,'Open'] - df_serie.at[index,'Low'] 
    elif df_serie.at[index,'Open'] > df_serie.at[index,'Close']:
        wick_size=df_serie.at[index,'Close'] - df_serie.at[index,'Low']
    df_serie.loc[index,'Shadow']=wick_size    
 

df_serie['Body'] = df_serie['Body'] / df_serie['Close']*100   
df_serie['Wick'] = df_serie['Wick'] / df_serie['Close']*100 
df_serie['Shadow'] = df_serie['Shadow'] / df_serie['Close']*100 

###################################################################################################3
## GRAFICI DI COMPARAZIONE DELLA DISTRIBUZIONE DELLE CARATTERISTICHE BODY, WICK E SHADOW

#DISTRIBUZIONE DEI BODI NELLE CANDELE DI MINIMO
bodyGraph=df_L2['Body'].plot.hist(title='body size',bins=300,alpha=0.5,color='blue',normed=True)   
bodyGraph2=df_serie['Body'].plot.hist(title='body size',bins=300,alpha=0.2,color='red', normed=True) 




bodyGraph=df_H2['Body'].plot.hist(title='body size',bins=300,alpha=0.5,color='blue',normed=True)   
bodyGraph2=df_serie['Body'].plot.hist(title='body size',bins=300,alpha=0.2,color='red', normed=True)  

Scatter_Wick_body=df_H2.plot.scatter(x='Body',y='Body_n',alpha=0.5)
Scatter_Wick_body=df_serie.plot.scatter(x='Body',y='Wick',alpha=0.5)


#dataframe di massimi e minimi con indice resettato non per visualizzazione grafica
#df_H=df_H.reset_index(drop=True)
#df_L=df_L.reset_index(drop=True)  
    
# #dataframe di frequenza highs
# df_orariH = df_H['Time'].dt.hour
# print (df_orariH)

# #dataframe di frequenza lows
# df_orariL = df_L['Time'].dt.hour
# print (df_orariL)

# #distribuzione oraria h
# ax = df_orariH.plot.hist(title='distribuzione oraria dei massimi',bins=24,alpha=0.5,color='red')

# #distribuzione oraria l
# bx = df_orariL.plot.hist(title='distribuzione oraria dei minimi',bins=24,alpha=0.5,color='blue', subplots=True)


   
# #dataframe massimi
# df_massimi = df_H['High']


# #dataframe di frequenza lows
# df_minimi = df_L['Low']


# #distribuzione oraria h
# cx = df_massimi.plot.hist(title='distribuzione valori massimi',bins=1000,alpha=0.5,color='red')

# #distribuzione oraria l
# dx = df_minimi.plot.hist(title='distribuzione valori minimi',bins=1000,alpha=0.5,color='blue', subplots=True)

 
# #dataframe massimi
# df_volumiH = df_H['Volume']


# #dataframe di frequenza lows
# df_volumiL = df_H['Volume']


# #distribuzione oraria h
# fx = df_volumiL.plot.hist(title='distribuzione volumi',bins=500,alpha=0.5,color='red')

#x=df_L.groupby(df["Time"].dt.hour().count().plot((kind="bar"))
             
#print(df["Time"].dt.hour())             
