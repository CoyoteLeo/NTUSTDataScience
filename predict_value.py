import numpy as np
import pandas as pd
from keras import losses
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, LeakyReLU
from keras.models import Sequential
from sklearn import preprocessing


def Model(attribute_num):
    model = Sequential()
    model.add(Dense(256, kernel_initializer='random_uniform', input_shape=(attribute_num,),
                    activity_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(128, kernel_initializer='random_uniform', activity_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(64, kernel_initializer='random_uniform', activity_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(32, kernel_initializer='random_uniform', activity_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=.001))
    # model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer=optimizers.Adam(lr=1e-1),
                  loss=losses.mean_absolute_error)
    return model


def loadData():
    train = pd.read_csv('train.csv', encoding="ISO-8859-1")
    x_data = train.drop(['Wage', 'Value', 'ID'], axis=1).values
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))  # Normalize
    x_normal = minmax_scale.fit_transform(x_data)

    wage = train['Wage'].values
    value = train['Value'].values

    new_x = []
    new_wage = []
    new_value = []
    index = np.random.permutation(range(x_data.shape[0]))
    for i in index:
        new_x.append(x_normal[i])
        new_wage.append(wage[i])
        new_value.append(value[i])

    return np.array(new_x), np.array(new_wage), np.array(new_value)


def main():
    train, wage, value = loadData()

    model = Model(train.shape[1])
    model.summary()

    save_model_path = 'model_value.h5'
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

    ans = model.predict(train)


if __name__ == '__main__':
    main()
