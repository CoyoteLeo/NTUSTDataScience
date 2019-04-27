import pandas as pd
from sklearn import preprocessing
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Flatten, Activation, Dropout, CuDNNGRU, BatchNormalization, Input
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 計算分數中的+號
def score(x):
    temp = []
    for i in x:
        if isinstance(i,str) and '+' in i:
            temp.append(sum(map(int, i.split('+'))))
        else:
            temp.append(i)
    return np.array(temp)

# 把'month-'去掉
def stripMonth(x):
    temp = []
    for i in x:
        if isinstance(i,str):
            temp.append(int(''.join(c for c in i if c.isdigit())))
        else:
            temp.append(i)
    return np.array(temp)

# preprocessing
'''
# 讀取薪資、市場價值的data, 並去掉錢幣符號
csv = pd.read_csv('CompleteDataset2019.csv')
wage =  csv.loc[:, ['ID', 'Wage']].set_index('ID')
wage['Wage'] = wage['Wage'].map(lambda x: float(x.strip('€K ')) * 1000 if 'K' in x else(float(x.strip('€M '))*100000) if 'M' in x else (x.strip('€ ')))
# print(wage)
value =  csv.loc[:, ['ID', 'Value']].set_index('ID')
value = value['Value'].map(lambda x: float(x.strip('€K ')) * 1000 if 'K' in x else(float(x.strip('€M '))*100000) if 'M' in x else (x.strip('€ ')))
label = pd.concat([wage, value], axis = 1)

# 讀取屬性資料
csv = pd.read_csv('CompleteDataset.csv', encoding = "ISO-8859-1")
temp = csv.iloc[:,0:47]
temp = temp.drop(['Name', 'Photo', 'Flag', 'Club Logo', 'Value','Wage', 'Unnamed: 0'], axis = 1)
temp = temp.apply(lambda x: score(x))

# 補上ID欄位
ID = csv.iloc[:,52]
temp = pd.concat([temp, ID], axis = 1)

# 補上prefer pos欄位並做one hot encoding
prefer = csv.iloc[:,63]
prefer = prefer.T.squeeze().str.split(' ', expand=True).stack()
prefer = pd.get_dummies(prefer).groupby(level=0).sum().drop([''], axis = 1)
df18 = pd.concat([temp,prefer], axis = 1)
df18_OneHot = pd.get_dummies(data = df18, columns = ['Nationality', 'Club', ])
df18_int = df18_OneHot.apply(lambda x: stripMonth(x)).set_index('ID')
train = pd.merge(label, df18_int, on=['ID'])

# 存data
train.to_csv('train.csv')
'''
# 讀取data
train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
x_data = train.drop(['Wage', 'Value'], axis = 1).values
minmax_scale = preprocessing.MinMaxScaler(feature_range = (0.0, 1.0))
x_normal = minmax_scale.fit_transform(x_data)

wage = train['Wage'].values
value = train['Value'].values

# 兜model
model = Sequential()
model.add(Dense(1024, input_shape=(x_normal.shape[1], )))
model.add(Activation('relu'))
model.add(Dense(4096)) 
model.add(Activation('relu'))
model.add(Dense(1024)) 
model.add(Activation('relu'))
model.add(Dense(1, activation='relu')) 
    
model.compile(optimizer='adam',
                loss='logcosh')

model.summary()
save_model_path = 'model.h5'
checkpoint = ModelCheckpoint(filepath = save_model_path, 
                                monitor='val_loss', 
                                save_weights_only=True, 
                                save_best_only=True, 
                                period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.1, 
                                patience = 3, 
                                verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', 
                                patience = 10, 
                                verbose=1)

history = model.fit(x=x_normal, y=wage, batch_size=128, epochs=1000, validation_split=0.1, callbacks = [checkpoint, reduce_lr, early_stopping])