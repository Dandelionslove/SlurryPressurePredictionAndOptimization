# coding:utf-8

from tensorflow import keras
from functions import *
from sklearn.model_selection import cross_validate
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from sklearn.model_selection import GridSearchCV
from keras.callbacks import LearningRateScheduler, EarlyStopping

np.random.seed(7)


class BPNN:
    def __init__(self, n_feature, model_name):
        self.input_length = n_feature
        # self.hidden_layer_dims = hidden_layer_dims
        self.model_name = model_name
        self.model = None
        self.history = None

    def construct_model(self,
                        # hidden_layer_num=None,
                        # neurons=None,
                        optimizer='Adam',
                        learning_rate=0.1,
                        decay_rate=0.8,
                        init_mode='uniform',
                        activation='relu',
                        drop_rate=0.5):
        model = keras.models.Sequential()
        # input layer
        model.add(keras.layers.Input(shape=(self.input_length,)))
        # hidden layers
        # if hidden_layer_dims is None:
        hidden_layer_dims = [16, 32, 64, 64, 48, 32]
        # hidden_layer_dims = [64, 128, 128, 256, 256, 128, 128, 64]
        # for dim in self.hidden_layer_dims:
        # hidden_layer_dims = [16, 32, 64, 96, 48, 8]
        # for neurons in [neurons1, neurons2, neurons3, neurons4, neurons5]:
        for neurons in hidden_layer_dims:
            model.add(keras.layers.Dense(neurons,
                                         activation=activation,
                                         kernel_initializer=init_mode,
                                         # kernel_constraint=keras.constraints.max_norm(weight_constraint),
                                         use_bias=True,
                                         bias_initializer=keras.initializers.Constant(0.1)
                                         ))
            model.add(keras.layers.Dropout(drop_rate))
        # output layer
        model.add(keras.layers.Dense(1))
        model.summary()
        model.compile(optimizer=get_optimizer(optimizer, learning_rate, decay_rate=decay_rate), loss='mse',
                      metrics=['mse', 'mae'])
        return model

    def grid_search_cv(self, X, y):
        model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=self.construct_model, verbose=1)

        init_mode = ['uniform', 'zero', 'glorot_normal']
        epochs = [50]
        batch_size = [32, 64, 128, 256]
        optimizer = ['SGD', 'RMSprop', 'Adam']
        decay_rate = np.arange(0.3, 0.9, 0.1)
        learning_rate = np.arange(0.01, 0.3, 0.01)
        activation = ['relu', 'sigmoid', 'tanh']
        drop_rate = np.arange(0.0, 0.5, 0.1)

        param_grid = dict(epochs=epochs, batch_size=batch_size,
                          # callbacks=call_backs,
                          optimizer=optimizer, activation=activation,
                          init_mode=init_mode, drop_rate=drop_rate,
                          decay_rate=decay_rate, learning_rate=learning_rate)
        # neurons1=neurons, neurons2=neurons,
        # neurons3=neurons, neurons4=neurons, neurons5=neurons)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFOLD_NUM, n_jobs=1, verbose=1,
                            scoring='neg_mean_squared_error')
        grid_result = grid.fit(X, y)
        best_score = np.abs(grid_result.best_score_)
        best_params = grid_result.best_params_
        result = {
            'cv': KFOLD_NUM,
            'best score': best_score,
            'best params': best_params
        }
        gscv_result_file_name = '{}.txt'.format(self.model_name)
        gscv_result_file = os.path.join(sub_gscv_result_dir, gscv_result_file_name)
        print_write_infos(gscv_result_file, 'w', **result)
        return best_params

    def train(self, train_X, train_y, is_save=True, is_plot_loss=True, **params):
        model = self.construct_model(optimizer=params['optimizer'], learning_rate=params['learning_rate'],
                                     decay_rate=params['decay_rate'], init_mode=params['init_mode'],
                                     activation=params['activation'],
                                     drop_rate=params['drop_rate'])
        model, history = nn_model_train(model, self.model_name, train_X, train_y,
                                        epochs=params['epochs'], batch_size=params['batch_size'],
                                        is_save=is_save, is_plot_loss=is_plot_loss)
        self.model = model
        self.history = history.history
        return model

    def cross_train(self, train_X, train_y, is_save=True, is_plot=True, **params):
        estimator = keras.wrappers.scikit_learn.KerasRegressor(build_fn=self.construct_model,
                                                               epochs=params['epochs'],
                                                               batch_size=params['batch_size'],
                                                               verbose=1)
        scores = cross_validate(estimator, train_X, train_y, cv=KFOLD_NUM,
                                scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                                return_train_score=True)
        val_r2 = np.mean(scores['test_r2'])
        val_mse = np.mean(scores['test_neg_mean_squared_error'])
        val_mae = np.mean(scores['test_neg_mean_absolute_error'])
        print('val r2:', scores['test_r2'], val_r2)
        print('val neg mse', scores['test_neg_mean_squared_error'], val_mse)
        print('val neg mae', scores['test_neg_mean_absolute_error'], val_mae)
        callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = estimator.fit(train_X, train_y,
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE,
                                verbose=1,
                                callbacks=[callback])
        if is_save:
            model_file_name = '{}.h5'.format(self.model_name)
            hist_file_name = '{}_hist.pickle'.format(self.model_name)
            model_file = os.path.join(sub_model_dir, model_file_name)
            hist_file = os.path.join(sub_model_dir, hist_file_name)
            estimator.model.save(model_file)
            with open(hist_file, 'wb') as f:
                pickle.dump(history.history, f)
        if is_plot:
            train_loss = history.history['loss']
            plot_loss(self.model_name, train_loss)
        self.model = estimator.model
        self.history = history.history
        return estimator.model

    def test(self, test_X, test_y, y_scaler, is_plot=True):
        nn_model_test(self.model, self.model_name, test_X, test_y, y_scaler, is_plot=is_plot)

    def load_model(self, model_file, hist_file=None):
        model = keras.models.load_model(model_file)
        self.model = model
        if hist_file is not None:
            with open(hist_file, 'rb') as f:
                history = pickle.load(f)
                self.history = history


def pressure_prediction_bpnn(mode=0):
    '''
    :param mode: 训练数据模式，0--非断层数据，1--断层数据，2--断层和非断层一起的数据
    :return:
    '''
    print('\033[1;35m {}, mode: {} \033[0m'.format('bpnn', mode))
    y_scaler, train_X, train_y, test_X, test_y = data_preparation(is_time_series=False, mode=mode)
    n_feature = train_X.shape[1]

    # hidden_layer_dims = [16, 64, 128, 48, 4]
    model_name = 'bpnn{}'.format(mode)
    bpnn = BPNN(n_feature, model_name)
    best_params = bpnn.grid_search_cv(train_X, train_y)
    bpnn.train(train_X, train_y, is_save=True, is_plot_loss=True, **best_params)
    # bpnn.cross_train(train_X, train_y)
    bpnn.test(test_X, test_y, y_scaler, is_plot=True)


if __name__ == '__main__':
    set_gpu_memory()
    mode = 0
    pressure_prediction_bpnn(mode)
