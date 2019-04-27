def main():
    # 讀取data
    train = pd.read_csv('train.csv', encoding="ISO-8859-1")
    x_data = train.drop(['Wage', 'Value'], axis=1).values
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 正規畫
    x_normal = minmax_scale.fit_transform(x_data)

    wage = train['Wage'].values
    value = train['Value'].values

    # 兜model
    model = Sequential()
    model.add(Dense(1024, input_shape=(x_normal.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam',
                  loss='mse')

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

    history = model.fit(x=x_normal, y=wage, batch_size=128, epochs=1000, validation_split=0.1,
                        callbacks=[checkpoint, reduce_lr, early_stopping])


if __name__ == '__main__':
    main()
