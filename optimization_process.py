# coding:utf-8

import joblib
import pandas as pd
import numpy as np
from functions import data_preparation, get_scaler
import copy
import os


def samples_extraction(df_X, df_y):
    np.random.seed(10)
    total_ring_nums = df_X['100001'].values
    extracted_rings = np.random.choice(total_ring_nums, size=25, replace=False)
    selected_samples_X = []
    selected_samples_y = []
    # selected_rings = []
    for ring_num in extracted_rings:
        extracted_df_X = df_X[df_X['100001'] == ring_num]
        length = len(extracted_df_X)
        if length > 0:
            extracted_df_y = df_y[df_y['100001'] == ring_num]
            # selected_rings.append(ring_num)
            extracted_index = np.random.choice(range(length), 2)  # 随机选取该环的两个样本
            extracted_sample_X = extracted_df_X.iloc[extracted_index, :]
            extracted_sample_y = extracted_df_y.iloc[extracted_index, :]
            selected_samples_X.append(extracted_sample_X)
            selected_samples_y.append(extracted_sample_y)
    samples_X_df = pd.concat(selected_samples_X, axis=0)
    samples_y_df = pd.concat(selected_samples_y, axis=0)
    # return samples_X_df, samples_y_df, selected_rings
    return samples_X_df, samples_y_df, extracted_rings


def greedy_optimization(model, samples_X_values, x_scaler, y_scaler):
    P_fields = np.arange(4.0, 4.51, 0.1)  # 气垫仓压力  0.001
    Q1_fields = np.arange(1980, 2000.001, 1)  # 进浆流量 0.01
    Q2_fields = np.arange(1990, 2000.001, 1)  # 出浆流量 0.01

    ideal_P2 = 4.0
    # optimized_result_list = []
    result = {
        'opt_P1': [],
        'opt_Q1': [],
        'opt_Q2': [],
        'opt_pred_P2': []
        # 'original_values': []
    }
    i = 1
    for sample in samples_X_values:
        print('count sample', i)
        i += 1
        cur_sample = copy.deepcopy(sample)
        optimized_result = {
            'opt_P1': None,
            'opt_Q1': None,
            'opt_Q2': None,
            'opt_pred_P2': None
            # 'original_values': None
        }
        for s_q1 in Q1_fields:
            for s_q2 in Q2_fields:
                for s_p in P_fields:
                    cur_sample[-1] = s_p
                    cur_sample[-2] = s_q2
                    cur_sample[-3] = s_q1
                    cur_norm_pred = model.predict(cur_sample.reshape(1, -1))
                    cur_pred = y_scaler.inverse_transform(cur_norm_pred)[0]
                    if optimized_result['opt_Q1'] is None or np.abs(ideal_P2 - cur_pred) < np.abs(
                            ideal_P2 - optimized_result['opt_pred_P2']):
                        optimized_result['opt_P1'] = s_p
                        optimized_result['opt_Q1'] = s_q1
                        optimized_result['opt_Q2'] = s_q2
                        optimized_result['opt_pred_P2'] = cur_pred
                    elif np.abs(ideal_P2 - cur_pred) == np.abs(
                            ideal_P2 - optimized_result['opt_pred_P2']):
                        optimized_result['opt_P1'] = np.random.choice([optimized_result['opt_P1'], s_p])
                        optimized_result['opt_pred_P2'] = np.random.choice([optimized_result['opt_pred_P2'], cur_pred])
                        # optimized_result['opt_Q1'] = np.random.choice([])

                    # elif np.abs(ideal_P2 - cur_pred) < np.abs(
                    #         ideal_P2 - optimized_result['opt_pred_P2']):
                    #     optimized_result['opt_P1'] = s_p
                    #     optimized_result['opt_Q1'] = s_q1
                    #     optimized_result['opt_Q2'] = s_q2
                    #     optimized_result['opt_pred_P2'] = cur_pred
                    # elif np.abs(ideal_P2 - cur_pred) == np.abs(
                    #         ideal_P2 - optimized_result['opt_pred_P2']):
                    #     r = np.random.choice([1, 2])
                    #     if r == 1:
                    #         optimized_result['opt_P1'] = s_p
                    #         optimized_result['opt_Q1'] = s_q1
                    #         optimized_result['opt_Q2'] = s_q2
                    #         optimized_result['opt_pred_P2'] = cur_pred
                        # optimized_result['original_values'] = list(x_scaler.inverse_transform(sample))
                    # print('cur sample:', cur_sample)
                    # print('cur pred', cur_pred)
        # optimized_result_list.append(optimized_result)
        print('*' * 16)
        print('opt result:', optimized_result)
        for key in result.keys():
            result[key].append(optimized_result[key])
    return result


if __name__ == '__main__':
    model_path = 'models/t60_s5_k6/adaboost0.m'
    X_data_file = 'preprocessed_data/t60_s5/pressure_X_mean_var_t60_s5.csv'
    y_data_file = 'preprocessed_data/t60_s5/pressure_y_mean_t60_s5.csv'
    df_X = pd.read_csv(X_data_file, sep=',', header=0, index_col=None)
    df_y = pd.read_csv(y_data_file, sep=',', header=0, index_col=None)
    X_names = ['100005_mean', '100002_mean', '100004_mean', '100003_mean',
               '100395_mean',
               '101403_mean', '101404_mean',
               '101465_mean']
    y_names = ['101419_mean']
    x_array, y_array, x_scaler, y_scaler = get_scaler(df_X[X_names], df_y[y_names])
    samples_X_df, samples_y_df, samples_rings = samples_extraction(df_X, df_y)
    samples_X_values = samples_X_df[X_names].values
    norm_X_values = x_scaler.transform(samples_X_values)

    model = joblib.load(model_path)

    optimized_result = greedy_optimization(model, samples_X_values, x_scaler, y_scaler)
    opt_df = pd.DataFrame(optimized_result, index=None)

    # file saving
    file_dir = 'optimization_result/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    opt_result_file = os.path.join(file_dir, 'opt_result.csv')
    sample_X_file = os.path.join(file_dir, 'samples_X.csv')
    sample_y_file = os.path.join(file_dir, 'samples_y.csv')
    opt_df.to_csv(opt_result_file, index=None)
    samples_X_df.to_csv(sample_X_file, index=None)
    samples_y_df.to_csv(sample_y_file, index=None)

