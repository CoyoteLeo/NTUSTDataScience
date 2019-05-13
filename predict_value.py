from preprocess import preprocess
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Activation, BatchNormalization, Conv1D, CuDNNGRU, Dense, Dropout, Embedding, Flatten, Input, MaxPooling1D
from keras.models import Sequential
from sklearn import preprocessing
import pandas as pd

def Model(attribute_num):
    model = Sequential()
    model.add(Dense(1024, input_shape=(attribute_num,)))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam',
                  loss='mse')
    return model

def loadData():
    train = pd.read_csv('train.csv', encoding="ISO-8859-1")
    x_data = train.drop(['Wage', 'Value'], axis=1).values
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))  # Normalize
    x_normal = minmax_scale.fit_transform(x_data)

    wage = train['Wage'].values
    value = train['Value'].values

    return x_normal, wage, value

def main():

    train, wage, value = loadData()

    model = Model(train.shape[1])
    model.summary()

    save_model_path = 'model.h5'
    checkpoint = ModelCheckpoint(filepath=save_model_path,
                                 monitor='val_loss',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=3,
                                  verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   verbose=1)

    history = model.fit(x=train, y=value, batch_size=128, epochs=1000, validation_split=0.1,
                        callbacks=[checkpoint, reduce_lr, early_stopping])


if __name__ == '__main__':
    main()
