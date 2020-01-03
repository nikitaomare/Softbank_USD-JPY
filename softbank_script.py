#importing library
import pandas as pd
from keras import models
from keras import layers
from keras.datasets import boston_housing
from keras.models import Model
from sklearn.model_selection import cross_val_score
from keras.layers import Input, SimpleRNN, Dense,LSTM,GRU
from keras import optimizers
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras import losses
from keras import metrics
from  more_itertools import unique_everseen
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time as time
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
from tpot import TPOTRegressor
## Text feature extraction for single 'id' by taking hidden state of LSTM
"""NOTE-> Trained on kaggle-gpu total time taken to train->1 hours approx"""
tic =time.time()
k=0
arr=[]
l=[]
tmp_0_forcast=pd.DataFrame()
for i in range(300):
    try:
        add ='tmp_'+ str(i)+'.csv'
        print(add)
        tmp=pd.read_csv('/kaggle/input/'+add)
        df_tm=tmp.set_index('id')
        count=list(tmp.groupby(['id']).size())
        id_list=list(unique_everseen(list(df_tm.index)))# to maintain sequence 
        new=dict(zip(id_list,count))
        a=0
        tmp_train=[]
        for i in id_list:
            n=new[i]
            b=a+n
            tmp_train.append(tmp[a:b])
            a=b
   
        predg=[]
        for j in range(len(tmp_train)):
            # define model
            a1=tmp_train[j].drop(['id'],axis=1)
            a = np.array(a1.values.reshape(1,a1.shape[1],a1.shape[0]))
            inputs1 = Input(shape=(a1.shape[1],a1.shape[0]))
            lstm1, state_h,state_c = LSTM(1, return_sequences=True, return_state=True)(inputs1)
            model = Model(inputs=inputs1, outputs=[lstm1, state_h,state_c])
            #print(model.predict(a))
            pred,q,e= model.predict(a,verbose=1)
            predg.append(pred)
        # define input data
        predgg=np.array(predg).reshape(len(tmp_train),300)
        predgg = pd.DataFrame(predgg)
        tmp_forcast=predgg.set_index([id_list])
        
        frames = [tmp_forcast,tmp_0_forcast]
        tmp_0_forcast=pd.concat(frames)
        print(tmp_0_forcast.shape)
    except:
        l.append(add)
        continue
toc =time.time()
print("took time in loading 241 text features by extracting hidden state of LSTM "+str((toc-tic))+" sec")
tmp_0_forcast.index.name = 'id'
tmp_0_forcast.to_csv('final_lstm.csv')
print("final_lstm shape"+str(final_lstm.shape))

## Combining text features and training data
final=pd.read_csv('final_lstm.csv')
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
final=final.set_index('id')
train=train.set_index('id')
test=test.set_index('id')
s1=set(final.index)
s2=set(train.index)
s3=set(test.index)

### Checking if all the 'id' in train are present in final_lstm
not_in_index=list()
for i in s2:
    if i not in s1:
        not_in_index.append(i)
## Removing 'id' which are not present in train
train=train.drop(not_in_index,axis=0)
## Concating both the datasets
l_train=list(train.index)
new_final=pd.DataFrame(columns=list(final.columns))
for i in l_train:
    new_final=new_final.append(final[final.index==i])
traineco=pd.concat([train,new_final],axis=1)
#only conataing id that are present in training dataset, in same order as id's in training dataset
## Same for the test 
not_in_index=list()
for i in s3:
    if i not in s1:
        not_in_index.append(i)
print("id not in final_lstm: "+str(not_in_index))
l_train=list(test.index)
new_final=pd.DataFrame(columns=list(final.columns))
for i in l_train:
    new_final=new_final.append(final[final.index==i])
testeco=pd.concat([test,new_final],axis=1)
testeco.to_csv('testeco_lstm.csv')
print("test data after combining :"+str(testeco.shape))


#Now train the model
test= pd.read_csv("testeco_lstm.csv")
train = pd.read_csv("traineco_lstm.csv")
gg=train.fillna(train.median())
y=gg['target']
X=gg.drop(['id','target'],axis=1)
print("X_shape:"+str(X.shape)," , y_shape :"+str(y.shape))
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import ExtraTreesRegressor
extra_tree = ExtraTreesRegressor(n_estimators=500,random_state=1234)
extra_tree.fit(X_train, y_train) 
ypredictions = extra_tree.predict(X_cv)
print(" Root Mean Absolute Error : ",sqrt(mean_squared_error(ypredictions, y_cv)))
extra_tree.fit(X, y) 
test2=test.drop(['id'],axis=1)
test2=test2.fillna(test2.median())
predictions =  extra_tree.predict(test2)
pred=pd.DataFrame(predictions) 
pred=pred.set_index([test['id']])
pred.to_csv("extra_tree_500.csv")

#Our best submission is extra_tree_500 giving accuracy-> 0.98098 on leaderboard,By Default ExtraTreesRegressor (n_estimators=500,random_state=1234)