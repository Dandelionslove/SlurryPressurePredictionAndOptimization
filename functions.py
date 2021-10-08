# coding:utf-8

from keras import backend as BK
import pandas as pd
import os
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# import matplotlib
# matplotlib.use('Qt5Agg')

EPOCHS = 10
BATCH_SIZE = 256
TIME_LENGTH = 60
STEP = 5
TEST_SIZE = 0.1
VALIDATION_SPLIT = 0.1

MIN_CV = 2
MAX_CV = 10
KFOLD_NUM = 6  # cur cv


# MODEL_SUFFIX = '_t{}_s{}_k{}_bs{}'.format(TIME_LENGTH, STEP, KFOLD_NUM, BATCH_SIZE)


def make_dir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


# dirs
sub_dir_name = 't{}_s{}_k{}'.format(TIME_LENGTH, STEP, KFOLD_NUM)
test_result_dir = 'test_results'
sub_test_result_dir = os.path.join(test_result_dir, sub_dir_name)
gscv_result_dir = 'gscv_results'
sub_gscv_result_dir = os.path.join(gscv_result_dir, sub_dir_name)
fig_dir = 'figs'
sub_fig_dir = os.path.join(fig_dir, sub_dir_name)
model_dir = 'models'
sub_model_dir = os.path.join(model_dir, sub_dir_name)

make_dir(test_result_dir)
make_dir(sub_test_result_dir)
make_dir(gscv_result_dir)
make_dir(sub_gscv_result_dir)
make_dir(fig_dir)
make_dir(sub_fig_dir)
make_dir(model_dir)
make_dir(sub_model_dir)


def print_write_infos(result_file, write_mode='w', **kwargs):
    with open(result_file, write_mode) as f:
        for key in kwargs.keys():
            info = '%s:\t%s' % (key, kwargs[key])
            print(info)
            f.write(info + '\n')
    f.close()


def plot_fitting_result(test_y_org, pred_y_org, model_name):
    test_y_array = test_y_org.reshape(-1)
    pred_y_array = pred_y_org.reshape(-1)
    diff = (test_y_org - pred_y_org).reshape(-1)

    fitting_fig_1 = os.path.join(sub_fig_dir, '{}_fitting1.png'.format(model_name))
    fitting_fig_2 = os.path.join(sub_fig_dir, '{}_fitting2.png'.format(model_name))
    diff_fig = os.path.join(sub_fig_dir, '{}_diff.png'.format(model_name))

    length = len(test_y_array)
    if length > 151:
        length = 151

    index = [i for i in range(1, length + 1, 5)]

    # fitting_1
    plt.figure(figsize=(12, 8))
    plt.plot(range(length), test_y_array[:length], 'mediumturquoise')
    plt.plot(range(length), pred_y_array[:length], 'lightcoral')
    plt.plot(range(length), test_y_array[:length], 'o', markersize=1, c='darkgreen', label='true')
    plt.plot(range(length), pred_y_array[:length], '^', markersize=1, c='brown', label='prediction')
    plt.title("fitting result on test data")
    plt.xticks(index)
    plt.legend(loc='upper right')
    plt.xlabel('The number of samples')
    plt.ylabel('Pressure (Bar)')
    plt.savefig(fitting_fig_1)
    # plt.show()

    # fitting2
    plt.figure(figsize=(12, 8))
    plt.plot(test_y_array[:length], test_y_array[:length], 'mediumturquoise', label='true')
    plt.scatter(test_y_array[:length], pred_y_array[:length], s=4, c='lightcoral', alpha=0.8, label='prediction')
    plt.title('fitting result on test data')
    plt.legend(loc='upper right')
    plt.xlabel('The number of samples')
    plt.ylabel('Pressure (Bar)')
    plt.savefig(fitting_fig_2)
    # plt.show()

    # plot diff
    plt.figure(figsize=(12, 8))
    plt.bar(range(length), diff[:length], fc='gray')
    plt.title('The difference between the true values and the predicted values')
    plt.xticks(index)
    plt.xlabel('The number of samples')
    plt.ylabel('Pressure (Bar)')
    plt.savefig(diff_fig)


def plot_loss(model_name, train_loss, val_loss=None):
    loss_fig_file_name = '{}_loss.png'.format(model_name)
    loss_fig_file = os.path.join(sub_fig_dir, loss_fig_file_name)
    plt.figure(figsize=(12, 8))
    index = range(1, len(train_loss) + 1)
    plt.plot(index, train_loss, color='red', label='training loss')
    if val_loss is not None:
        plt.plot(index, val_loss, color='green', label='validation loss')
    plt.legend(loc='best')
    plt.title('Loss in the training process')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.xticks(index)
    plt.savefig(loss_fig_file)


def get_optimizer(optimizer, lr, decay_rate=0.85):
    if optimizer == 'Adagrad':
        opt = keras.optimizers.Adagrad(learning_rate=lr, decay=decay_rate)
    elif optimizer == 'Adadelta':
        opt = keras.optimizers.Adadelta(learning_rate=lr, decay=decay_rate)
    elif optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=lr, decay=decay_rate)
    else:
        opt = keras.optimizers.Adam(learning_rate=lr, decay=decay_rate)
    return opt


def set_gpu_memory():
    # set GPU memory
    if 'tensorflow' == BK.backend():
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)


def save_model_and_history(model, model_name, history):
    model_file = sub_model_dir + '/{}.h5'.format(model_name)
    model.save(model_file)
    history_file = sub_model_dir + '/{}_history.'.format(model_name)
    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)


def nn_model_train(model, model_name, train_X, train_y, epochs=20, batch_size=64, is_save=True, is_plot_loss=True):
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(train_X, train_y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=VALIDATION_SPLIT,
                        verbose=1,
                        callbacks=[early_stopping])
    if is_save:
        model_file_name = '{}.h5'.format(model_name)
        hist_file_name = '{}_hist.pickle'.format(model_name)
        model_file = os.path.join(sub_model_dir, model_file_name)
        hist_file = os.path.join(sub_model_dir, hist_file_name)
        model.save(model_file)
        with open(hist_file, 'wb') as f:
            pickle.dump(history.history, f)
    if is_plot_loss:
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        plot_loss(model_name, train_loss, val_loss)
    return model, history


def nn_model_test(model, model_name, test_X, test_y, y_scaler, is_plot=True):
    pred_y = model.predict(test_X)
    test_r2 = r2_score(test_y, pred_y)
    test_mse = mean_squared_error(test_y, pred_y)
    test_mae = mean_absolute_error(test_y, pred_y)
    result = {
        'test r2': test_r2,
        'test mse': test_mse,
        'test mae': test_mae
    }
    test_y_org = y_scaler.inverse_transform(test_y)
    pred_y_org = y_scaler.inverse_transform(pred_y)
    test_result_file_name = '{}.txt'.format(model_name)
    test_pred_file_name = 'pred_test_{}.csv'.format(model_name)
    test_result_file = os.path.join(sub_test_result_dir, test_result_file_name)
    test_pred_file = os.path.join(sub_test_result_dir, test_pred_file_name)
    print_write_infos(test_result_file, 'w', **result)
    test_pred_df = pd.DataFrame({
        'test_y': list(test_y_org),
        'pred_y': list(pred_y_org)
    })
    test_pred_df.to_csv(test_pred_file, index=None)
    if is_plot:
        plot_fitting_result(test_y_org, pred_y_org, model_name)


def get_scaler(X_df, y_series, sample_num=None, time_length=None, input_dim=None, is_time_series=False):
    x_scaler = StandardScaler()
    x_scaler.fit(X_df)
    x_array = x_scaler.transform(X_df)
    if is_time_series:
        x_array = x_array.reshape((sample_num, time_length, input_dim))

    y_scaler = StandardScaler()
    y_original_array = y_series.values.reshape(-1, 1)
    y_scaler.fit(y_original_array)
    y_array = y_scaler.transform(y_original_array)

    print('x_array shape:\t', x_array.shape)
    print('y_array shape:\t', y_array.shape)

    return x_array, y_array, x_scaler, y_scaler


def data_construction(mode=0, is_time_series=False):
    name = 'mean_var'
    if is_time_series:
        name = 'time_series'
    if mode == 0:  # 非断层数据
        data_X_file = 'preprocessed_data/t{0}_s{1}/pressure_X_{2}_t{0}_s{1}.csv'.format(TIME_LENGTH, STEP, name)
        data_y_file = 'preprocessed_data/t{0}_s{1}/pressure_y_mean_t{0}_s{1}.csv'.format(TIME_LENGTH, STEP)
        df_X = pd.read_csv(data_X_file, sep=',', index_col=None, header=0)
        df_y = pd.read_csv(data_y_file, sep=',', index_col=None, header=0)
    elif mode == 1:  # 断层数据
        data_X_file = 'preprocessed_data/t{0}_s{1}/fault_pressure_X_{2}_t{0}_s{1}.csv'.format(TIME_LENGTH, STEP, name)
        data_y_file = 'preprocessed_data/t{0}_s{1}/fault_pressure_y_mean_t{0}_s{1}.csv'.format(TIME_LENGTH, STEP)
        df_X = pd.read_csv(data_X_file, sep=',', index_col=None, header=0)
        df_y = pd.read_csv(data_y_file, sep=',', index_col=None, header=0)
    else:
        data_X_file1 = 'preprocessed_data/t{0}_s{1}/pressure_X_{2}_t{0}_s{1}.csv'.format(TIME_LENGTH, STEP, name)
        data_y_file1 = 'preprocessed_data/t{0}_s{1}/pressure_y_mean_t{0}_s{1}.csv'.format(TIME_LENGTH, STEP)
        data_X_file2 = 'preprocessed_data/t{0}_s{1}/fault_pressure_X_{2}_t{0}_s{1}.csv'.format(TIME_LENGTH, STEP, name)
        data_y_file2 = 'preprocessed_data/t{0}_s{1}/fault_pressure_y_mean_t{0}_s{1}.csv'.format(TIME_LENGTH, STEP)
        df_X1 = pd.read_csv(data_X_file1, sep=',', index_col=None, header=0)
        df_y1 = pd.read_csv(data_y_file1, sep=',', index_col=None, header=0)
        df_X2 = pd.read_csv(data_X_file2, sep=',', index_col=None, header=0)
        df_y2 = pd.read_csv(data_y_file2, sep=',', index_col=None, header=0)
        df_X = pd.concat([df_X1, df_X2], axis=0)
        df_y = pd.concat([df_y1, df_y2], axis=0)
    print('the length of the samples is ', len(df_y))
    return df_X, df_y


def data_split(df_X, series_y, test_size=0.1, is_time_series=False):
    if is_time_series:
        input_dim = df_X.shape[1]
        sample_num = series_y.shape[0]
        x_array, y_array, x_scaler, y_scaler = get_scaler(df_X, series_y,
                                                          sample_num=sample_num,
                                                          time_length=TIME_LENGTH,
                                                          input_dim=input_dim,
                                                          is_time_series=True)
    else:
        # x_scaler = StandardScaler()
        # x_scaler.fit(df_X)
        # x_array = x_scaler.transform(df_X)
        # y_scaler = StandardScaler()
        # array_y = series_y.values.reshape(-1, 1)
        # y_scaler.fit(array_y)
        # y_array = y_scaler.transform(array_y)
        # y_array = y_array.reshape(-1)
        x_array, y_array, x_scaler, y_scaler = get_scaler(df_X, series_y)

    train_X, test_X, train_y, test_y = train_test_split(x_array, y_array, test_size=test_size, random_state=1)
    return y_scaler, train_X, train_y, test_X, test_y


'''
数据准备
'''


def data_preparation(is_time_series=False, mode=0):
    if not is_time_series:
        x_cols = ['100005_mean', '100002_mean', '100004_mean', '100003_mean', '100395_mean', '101403_mean', '101404_mean',
                  '101465_mean']
        y_col = ['101419_mean']
    else:
        x_cols = ['100005', '100004', '100003', '100002', '100395', '101465', '101403', '101404']
        y_col = ['101419_mean']
    df_X, df_y = data_construction(mode=mode, is_time_series=is_time_series)
    feature_df_X = df_X[x_cols]
    series_y = df_y[y_col]
    y_scaler, train_X, train_y, test_X, test_y = data_split(feature_df_X, series_y, test_size=TEST_SIZE,
                                                            is_time_series=is_time_series)
    return y_scaler, train_X, train_y, test_X, test_y


if __name__ == '__main__':
    file_dir = 'data_ring'
    preprocessed_file_dir = 'preprocessed_data'
