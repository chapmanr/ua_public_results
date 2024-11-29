import pandas as pd


import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt


from modules.kdd_libs import *
from modules.logger import logger

def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.InputLayer(shape=(699,)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(1, activation='sigmoid'))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  
  #keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

def basic_preprocessing(X, basic_only=False):
    X["brand"] = X["brand"].str.strip()
    X['brand'] = X['brand'].str.replace('M', 'm')
    X['brand'] = X['brand'].str.replace('marca', '')
    X["brand"] = X["brand"].str.strip()

    #SKU is an ID - therefore categorical not a "real" number
    X['sku'] = X['sku'].apply(lambda x: str(x))
    
    
    X[['new_pvp', 'discount']] = X['new_pvp (discount)'].str.split(' ', expand=True)
    X.drop(['new_pvp (discount)'], axis=1, inplace=True)

    X['oldpvp']     = X['oldpvp'].apply(lambda x: str(x).replace(',', '.'))
    #Prices should be numbers
    X['oldpvp']     = X['oldpvp'].apply(lambda x: float(x))

   
    X['new_pvp']    = X['new_pvp'].apply(lambda x: str(x).replace(',', '.'))
    #Prices should be numbers
    X['new_pvp']    = X['new_pvp'].apply(lambda x: float(x))


    X['discount'] = X['discount'].apply(lambda x: str(x).replace('%', ''))
    X['discount'] = X['discount'].apply(lambda x: str(x).replace('(', ''))
    X['discount'] = X['discount'].apply(lambda x: str(x).replace(')', ''))
    X['discount'] = X['discount'].apply(lambda x: '0.'+x if len(x) == 2 else x)

    X['discount'] = X['discount'].apply(lambda x: float(x))
    X["discount"] = X["discount"].apply(lambda x: x if x <= 1.0 else x / 100)

    #Weight should be a number - but I'm not sure how useful it is...
    X['weight (g)'] = X['weight (g)'].apply(lambda x: str(x).replace(' ', ''))
    most_frequ_weight = X['weight (g)'].mode()
    most_frequ_weight = most_frequ_weight[0]

    X['weight (g)'] = X['weight (g)'].apply(lambda x: int(x) if x != '' and x != 'nan' else int(most_frequ_weight))


    X['expiring_date'] = X['expiring_date'].apply(lambda x: str(x).replace('/', '-'))
    X['labelling_date'] = X['labelling_date'].apply(lambda x: str(x).replace('/', '-'))
    
    
    X['expiring_date'] = pd.to_datetime(X['expiring_date'], format='%d-%m-%Y')
    X['expiring_day'] = X['expiring_date'].dt.dayofweek
    X['expiring_day'] = X['expiring_day'].astype(str)

    X["idstore"] = X["idstore"].apply(str)       

    X['labelling_date'] = pd.to_datetime(X['labelling_date'], format='mixed')
    X["labelling_day"] = X["labelling_date"].dt.dayofweek  
    X["labelling_day"] = X["labelling_day"].astype(str)


    X['duration_days'] = pd.to_datetime(X['expiring_date']) - pd.to_datetime(X['labelling_date'])
    X['duration_days'] =  X['duration_days'].dt.total_seconds() / (24 * 60 * 60)

    # calculate total cost from profit and margin
    X_cost = ((100 - X['Margin (%)'].to_numpy()) / X['Margin (%)'].to_numpy()) * X['Profit (€)'].to_numpy()
    X.loc[:, ('Cost (€)')] = X_cost

    X.drop(['labelqty'], axis=1, inplace=True)
    X.drop(['labelling_date'], axis=1, inplace=True)
    X.drop(['expiring_date'], axis=1, inplace=True)
    
    #Which one of these buggers is now irrelevent?
    #X.drop(['new_pvp', 'oldpvp'], axis=1, inplace=True)

    X.drop(['Margin (%)'], axis=1, inplace=True)
    
    return X

def drop_high_corrs(X_tr, X_ts):
    X_tr.drop(['new_pvp'], axis=1, inplace=True)
    X_ts.drop(['new_pvp'], axis=1, inplace=True)
   
    X_tr.drop(['oldpvp'], axis=1, inplace=True)
    X_ts.drop(['oldpvp'], axis=1, inplace=True)
   
    X_tr.drop(['Cost (€)'], axis=1, inplace=True)
    X_ts.drop(['Cost (€)'], axis=1, inplace=True)

    return X_tr, X_ts


def test_data_transform():
    log = logger("results/dl_results.md")
    X_train, y, X_test = load_raw_data("clean_data/")

    X_train = basic_preprocessing(X_train, basic_only=False)
    X_test = basic_preprocessing(X_test, basic_only=False)

    drop_high_corrs(X_train, X_test)
    X_train.describe()

    clf = generate_full_proc_pipeline(X_train, log=log)

    X = clf.fit_transform(X_train)
    X_test = clf.transform(X_test)
    print("Transformed data shape " + str(X.shape) + " " + str(X_test.shape))    
    
    return X, y, X_test, log



def train_dl():
    X_trans, y, X_pred_trans, log = test_data_transform()

    
    #learn the categorical embeddings here and save them
    #X_train, X_pred = learn_categorical_embeddings(X_train, y_train, X_pred, log)

    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, stratify=y, test_size=0.2, random_state=42)
    
    tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
        """)
    # Build the model with the optimal hyperparameters and train it on the data
    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)

    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)

    eval_result = hypermodel.evaluate(X_test, y_test)
    print("[test loss, test accuracy]:", eval_result)


    hypermodel.save('models/dl_hyper_model.h5')
    model.save('models/dl_model.h5')



if __name__ == '__main__':
    train_the_dl = True
    if train_the_dl:
        train_dl()
    
    
    hyper_model = keras.models.load_model('models/dl_hyper_model.h5')
    print("Hyper Model type = " + str(type(hyper_model)))
    print(str(hyper_model.summary()))


    model = keras.models.load_model('models/dl_model.h5')
    print("Model type = " + str(type(model)))
    print(str(model.summary()))


    X, y, X_pred = load_raw_data('clean_data/')

    X = basic_preprocessing(X, basic_only=False)
    X_pred = basic_preprocessing(X_pred, basic_only=False)
    drop_high_corrs(X, X_pred)

    clf = generate_full_proc_pipeline(X, log=None)
    X = clf.fit_transform(X)
    X_pred_trans = clf.transform(X_pred)
    
    y_pred = hyper_model.predict(X_pred_trans)
    print("Predicted shape " + str(y_pred.shape))
    np_list = np.array(y_pred)
    print("Predicted values " + str(np_list))
    np_list = np_list
    print(str(y_pred))

    pred_df = pd.DataFrame(np_list, columns=['sold'], index=X_pred.index)
    pred_df.to_csv('results/y_pred_keras_hyper.csv')
    pred_df.describe().to_csv('results/y_pred_keras_hyper_stats.csv')

    pred_df[pred_df['sold'] > 0.5]  = 1.0
    pred_df[pred_df['sold'] <= 0.5] = 0.0
    pred_df.to_csv('results/y_pred_keras_hyper_binary.csv')
    pred_df.describe().to_csv('results/y_pred_keras_hyper_binary_stats.csv')

    #log.close()