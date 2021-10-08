# coding:utf-8

import pandas as pd
import os
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

TIME_LENGTH = 60  # 时间序列的长度, time_steps
# STEP = 5  # 步长
# MAX_T_OVER = int((TIME_LENGTH + 4) * 5)  # 容错时间差
STEP = 10  # 步长
MAX_T_OVER = int((TIME_LENGTH + 4) * 10)  # 容错时间差
TEST_SIZE = 0.2
IS_TEST = False  # 供测试使用

# t-时间
# wm-掘进状态
# 100001-环号
# 100002-推进速度，100003-刀盘转速，100005-总推力，100004-刀盘扭矩
# 100395-贯入度
# 101465-气垫仓压力，101403-进浆流量，101404-出浆流量
# 101419-泥水仓顶部压力
# 101876-刀盘总挤压力，100008-刀盘实际功率，101824-刀盘旋转功率
# 860424-气垫仓泥浆液位
# 101420-泥水仓中部压力

# 所有需要的列名，包括X和y的
extracted_cols = ['ID', 't', 'wm', '100001',
                  '101876',  # 刀盘总挤压力
                  '100008',  # 刀盘实际功率
                  '101824',  # 刀盘旋转功率
                  '100005',  # 总推进力
                  '100002',  # 推进速度
                  '100003',  # 刀盘转速
                  '100395',  # 贯入度
                  '101403',  # 进浆流量
                  '101404',  # 出浆流量
                  '101401',  # 进浆密度
                  '101402',  # 出浆密度
                  '860424',  # 泥浆液位值
                  '101465',  # 气垫仓压力
                  '101419',  # 泥水仓顶部压力
                  '101420',  # 泥水仓轴部压力
                  '860710',  # 总注浆量
                  '100471',  # 滚动角
                  '100470',  # 俯仰角
                  '100004'  # 刀盘扭矩
                  ]
# 环号+特征名
# param_feature_cols = ['100001', '101419', '100395', '100003', '100002']
param_feature_cols = ['100001',
                      '101876',  # 刀盘总挤压力
                      '100008',  # 刀盘实际功率
                      '101824',  # 刀盘旋转功率
                      '100005',  # 总推进力
                      '100004',  # 刀盘扭矩
                      '100002',  # 推进速度
                      '100003',  # 刀盘转速
                      '100395',  # 贯入度
                      '101403',  # 进浆流量
                      '101404',  # 出浆流量
                      '101401',  # 进浆密度
                      '101402',  # 出浆密度
                      '860424',  # 泥浆液位值
                      '101465',  # 气垫仓压力
                      '101419',  # 泥水仓顶部压力
                      '101420',  # 泥水仓轴部压力
                      '860710',  # 总注浆量
                      '100471',  # 滚动角
                      '100470'  # 俯仰角
                      ]
pressure_feature_cols = ['100001', '100005', '100003', '100004', '100002', '100395', '101465', '101403', '101404']

# 单独提出来的y列名
pressure_col = '101419'
f_col = '100005'
t_col = '100004'

# 主三级围岩环号
level3_rings = [i for i in range(856, 2126)]
# 主四级围岩环号
level4_rings = [i for i in range(291, 526)] + [i for i in range(726, 856)]
# 主五级围岩环号
level5_rings = [i for i in range(1, 291)] + [i for i in range(526, 726)]
# 断层破碎带环号
fault_rings = [i for i in range(364, 391)] + [i for i in range(507, 559)] \
              + [i for i in range(920, 953)] + [i for i in range(837, 888)]


# 异常值处理
def three_sigma(ser1):  # ser1表示传入DataFrame的某一列
    ser1 = pd.Series(ser1, dtype='float32')
    mean_value = ser1.mean(axis=0)  # 求平均值
    std_value = ser1.std(axis=0)  # 求标准差
    rule = (mean_value - 3 * std_value > ser1) | (ser1.mean() + 3 * ser1.std() < ser1)
    # 位于（u-3std,u+3std）区间的数据是正常的，不在这个区间的数据为异常的
    # 一旦发现有异常值，就标注为True，否则标注为False
    index = np.arange(ser1.shape[0])[rule]  # 返回异常值的位置索引
    outrange = ser1.iloc[index]  # 获取异常数据
    return index, outrange, mean_value, std_value


# 对原始数据进行预处理，包括去除空值和异常值
def data_preprocessed(df_drop_null):
    for col in extracted_cols[3:]:
        df_drop_null[col] = df_drop_null[col].astype('float32')

    # 3σ准则剔除异常值
    # 各个参数异常值显示
    v_index, v_outrange, v_mean, v_std = three_sigma(df_drop_null['100002'])  # 推进速度
    G_index, G_outrange, G_mean, G_std = three_sigma(df_drop_null['100395'])  # 贯入度
    J_index, J_outrange, J_mean, J_std = three_sigma(df_drop_null['101403'])  # 进浆管流量
    H_index, H_outrange, H_mean, H_std = three_sigma(df_drop_null['860424'])  # 气垫仓泥浆液位
    P_index, P_outrange, P_mean, P_std = three_sigma(df_drop_null['101465'])  # 气垫仓压力
    p2_index, p2_outrange, p2_mean, p2_std = three_sigma(df_drop_null['101420'])  # 泥水仓压力02
    p4_index, p4_outrange, p4_mean, p4_std = three_sigma(df_drop_null['101824'])  # 刀盘旋转功率
    F1_index, F1_outrange, F1_mean, F1_std = three_sigma(df_drop_null['101876'])  # 刀盘总挤压力
    F2_index, F2_outrange, F2_mean, F2_std = three_sigma(df_drop_null['100005'])  # 刀盘推力

    # 提取异常值的数据再剔除，剔除条件如下：
    # 推进速度大于50及小于等于0、贯入度小于等于0或大于30、
    # 进浆管流量异常及小于等于0、出浆管流量小于等于0、气垫仓泥浆液位全部异常值
    # 气垫仓压力异常及小于等于0、泥水仓压力01小于等于0、泥水仓压力02异常及小于0
    # 刀盘实际功率大于200及小于等于0，刀盘旋转功率异常及小于等于0,刀盘挤压力小于5000，刀盘推力小于15000

    data_outrange = np.concatenate(
        (v_index[v_outrange > 50], v_index[v_outrange <= 0], G_index[G_outrange <= 0], G_index[G_outrange > 30],
         J_index, np.arange(df_drop_null.shape[0])[df_drop_null['101403'] <= 0],
         np.arange(df_drop_null.shape[0])[df_drop_null['101404'] <= 0],
         H_index, P_index, np.arange(df_drop_null.shape[0])[df_drop_null['101465'] <= 0],
         np.arange(df_drop_null.shape[0])[df_drop_null['101419'] <= 0], p2_index,
         np.arange(df_drop_null.shape[0])[df_drop_null['101420'] <= 0],
         np.arange(df_drop_null.shape[0])[df_drop_null['100008'] <= 0],
         np.arange(df_drop_null.shape[0])[df_drop_null['100008'] > 200],
         p4_index, np.arange(df_drop_null.shape[0])[df_drop_null['101824'] <= 0],
         F1_index[F1_outrange < 5000], F2_index[F2_outrange < 15000]), axis=0)
    data_outrange_unique = np.unique(data_outrange)  # 去除重复元素

    # 修改行名
    df_drop_null.index = np.arange(df_drop_null.shape[0])
    # 删除异常行
    df_normal = df_drop_null.drop(data_outrange_unique)  # 删除异常行
    df_normal.index = np.arange(df_normal.shape[0])  # 行索引重命名

    return df_normal


# 构造时序特征数据
def data_construct(file_dir, preprocessed_file_dir, is_fault=False):
    files = os.listdir(file_dir)
    pressure_X_time_df_list = []
    pressure_X_mean_var_df_list = []
    pressure_y_mean_df_list = []
    param_X_time_df_list = []
    param_X_mean_var_df_list = []
    param_y_mean_df_list = []

    for file in files:
        if not os.path.isdir(file):
            print('file name: %s' % file)
            ring_num = int(file.split('.')[0])
            if is_fault:  # 断层数据
                if ring_num not in fault_rings:
                    print('跳过环号%d--非断层破碎带地层' % ring_num)
                    continue
            else:  # 非断层数据
                if ring_num in fault_rings:
                    print('跳过环号%d--断层破碎带地层' % ring_num)
                    continue
            file_name = file_dir + '/' + file
            df = pd.read_csv(file_name, sep=',', index_col=0, header=0)
            df = df[(df['100006'] > 0) | (df['wm'] > 0)]
            # 提取需要的特征列
            df = df[extracted_cols]
            # 空值处理
            df_drop_null = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
            # 异常值处理
            df_normal = data_preprocessed(df_drop_null)
            if len(df_normal) == 0:
                continue
            ring_data_extracted_result = data_construct_for_ring(TIME_LENGTH,
                                                                 STEP,
                                                                 df_normal,
                                                                 max_t_over=MAX_T_OVER)
            if ring_data_extracted_result['pressure_X_time_df'] is not None:
                pressure_X_time_df_list.append(ring_data_extracted_result['pressure_X_time_df'])
                pressure_X_mean_var_df_list.append(ring_data_extracted_result['pressure_X_mean_var_df'])
                pressure_y_mean_df_list.append(ring_data_extracted_result['pressure_y_mean_df'])
            if ring_data_extracted_result['param_X_time_df'] is not None:
                param_X_time_df_list.append(ring_data_extracted_result['param_X_time_df'])
                param_X_mean_var_df_list.append(ring_data_extracted_result['param_X_mean_var_df'])
                param_y_mean_df_list.append(ring_data_extracted_result['param_y_mean_df'])
            # break
    fault_state = ''
    if is_fault:
        fault_state = 'fault_'
    data_file_dir = os.path.join(preprocessed_file_dir, 't%d_s%d' % (TIME_LENGTH, STEP))
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)
    pressure_X_time_file = '%s/%spressure_X_time_series_t%d_s%d.csv' % (
        data_file_dir, fault_state, TIME_LENGTH, STEP)
    pressure_X_mean_var_file = '%s/%spressure_X_mean_var_t%d_s%d.csv' % (
        data_file_dir, fault_state, TIME_LENGTH, STEP)
    pressure_y_mean_file = '%s/%spressure_y_mean_t%d_s%d.csv' % (
        data_file_dir, fault_state, TIME_LENGTH, STEP)
    param_X_time_file = '%s/%sparam_X_time_series_t%d_s%d.csv' % (
        data_file_dir, fault_state, TIME_LENGTH, STEP)
    param_X_mean_var_file = '%s/%sparam_X_mean_var_t%d_s%d.csv' % (
        data_file_dir, fault_state, TIME_LENGTH, STEP)
    param_y_mean_file = '%s/%sparam_y_mean_t%d_s%d.csv' % (
        data_file_dir, fault_state, TIME_LENGTH, STEP)
    if len(pressure_X_time_df_list) > 0:
        total_pressure_X_time_df = pd.concat(pressure_X_time_df_list, axis=0)
        total_pressure_X_mean_var_df = pd.concat(pressure_X_mean_var_df_list, axis=0)
        total_pressure_y_mean_df = pd.concat(pressure_y_mean_df_list, axis=0)
        total_pressure_X_time_df.to_csv(pressure_X_time_file, index=None)
        total_pressure_X_mean_var_df.to_csv(pressure_X_mean_var_file, index=None)
        total_pressure_y_mean_df.to_csv(pressure_y_mean_file, index=None)
        print('the number of the samples for pressure prediction is:\t %d' % len(total_pressure_y_mean_df))
        print('files are saved successfully.')
    if len(param_X_time_df_list) > 0:
        total_param_X_time_df = pd.concat(param_X_time_df_list, axis=0)
        total_param_X_mean_var_df = pd.concat(param_X_mean_var_df_list, axis=0)
        total_param_y_mean_df = pd.concat(param_y_mean_df_list, axis=0)
        total_param_X_time_df.to_csv(param_X_time_file, index=None)
        total_param_X_mean_var_df.to_csv(param_X_mean_var_file, index=None)
        total_param_y_mean_df.to_csv(param_y_mean_file, index=None)
        print('the number of the samples for param prediction is:\t %d' % len(total_param_y_mean_df))
        print('files are saved successfully.')
    return True


# 被调用，对特定环号进行时序特征数据的构造
def data_construct_for_ring(time_length, step, df, max_t_over=10):
    # time_length = f_length + pred_length
    t = df['t']
    ring_num = int(df['100001'].iloc[0])
    data_length = df.shape[0]
    # 存储泥水仓压力预测要用到的数据
    pressure_X_time_df_list = []
    pressure_X_mean_var_dict = {}
    pressure_y_mean_dict = {}
    # 存储操作参数预测要用到的数据
    param_X_time_df_list = []  # fpi, tpi + other time-series features
    param_X_mean_var_dict = {}
    param_y_mean_dict = {}

    for start_index in range(0, data_length, step):
        end_index = start_index + time_length
        if end_index >= data_length:
            print('超过数据长度，本环数据构造结束')
            break
        start_t = t.iloc[start_index]
        end_t = t.iloc[end_index]
        start_time = str_to_time(start_t)
        end_time = str_to_time(end_t)
        time_diff = (end_time - start_time).seconds
        # print('time diff:', time_diff)
        if int(time_diff) > max_t_over:  # 如果整个时间差大于容错时间差，则跳过该段
            print('时间差过大，跳过该段, 容错时间为%d, 本段时间差为%d' % (max_t_over, int(time_diff)))
            continue
        else:
            cur_part_df = df.iloc[start_index: end_index, :]
            # 泥水仓压力预测模型数据构造
            # 时序数据
            pressure_X_time_df = cur_part_df[pressure_feature_cols]
            pressure_X_time_df_list.append(pressure_X_time_df)
            # 非时序数据
            ring_name = '100001'
            if ring_name not in pressure_X_mean_var_dict.keys():
                pressure_X_mean_var_dict[ring_name] = []
            pressure_X_mean_var_dict[ring_name].append(ring_num)
            for x_col in pressure_feature_cols[1:]:
                col_mean_name = x_col + '_mean'
                col_var_name = x_col + '_var'
                if col_mean_name not in pressure_X_mean_var_dict.keys():
                    pressure_X_mean_var_dict[col_mean_name] = []
                    pressure_X_mean_var_dict[col_var_name] = []
                pressure_X_mean_var_dict[col_mean_name].append(pressure_X_time_df[x_col].mean())
                pressure_X_mean_var_dict[col_var_name].append(pressure_X_time_df[x_col].var())
            pressure_y_series = cur_part_df[pressure_col]
            if ring_name not in pressure_y_mean_dict.keys():
                pressure_y_mean_dict[ring_name] = []
            pressure_y_mean_dict[ring_name].append(ring_num)
            pressure_y_col_mean_name = pressure_col + '_mean'
            if pressure_y_col_mean_name not in pressure_y_mean_dict.keys():
                pressure_y_mean_dict[pressure_y_col_mean_name] = []
            pressure_y_mean_dict[pressure_y_col_mean_name].append(pressure_y_series.mean())

            # 岩机映射模型数据构造
            second_end_index = end_index + time_length
            if second_end_index >= data_length:
                print('超过数据长度，本环数据构造结束')
                break
            second_end_t = t.iloc[second_end_index]
            second_end_time = str_to_time(second_end_t)
            second_time_diff = (second_end_time - start_time).seconds
            if int(second_time_diff) > 2 * max_t_over:
                print('构造掘进参数预测模型数据时时间差过大，跳过该段, 容错时间为%d, 本段时间差为%d' % (max_t_over * 2, int(time_diff)))
                continue
            next_part_df = df.iloc[end_index: second_end_index, :]
            cur_fpi_series = cur_part_df['100005'] / (cur_part_df['100395'] * 1000)
            cur_tpi_series = cur_part_df['100004'] / cur_part_df['100395']
            # cur_param_X_time_df = pd.DataFrame(
            #     {'front_fpi': cur_fpi_series, 'front_tpi': cur_tpi_series},
            #     index=None)
            # 时序数据
            next_param_X_time_df = next_part_df[param_feature_cols]
            # param_X_time_df = pd.concat([cur_param_X_time_df, next_param_X_time_df], axis=1)
            param_X_time_df = next_param_X_time_df
            param_X_time_df['front_fpi'] = cur_fpi_series.values
            param_X_time_df['front_tpi'] = cur_tpi_series.values
            param_X_time_df_list.append(param_X_time_df)
            # 非时序数据
            if ring_name not in param_X_mean_var_dict.keys():
                param_X_mean_var_dict[ring_name] = []
            param_X_mean_var_dict[ring_name].append(ring_num)
            for x_col in param_feature_cols[1:]:
                col_mean_name = x_col + '_mean'
                col_var_name = x_col + '_var'
                if col_mean_name not in param_X_mean_var_dict.keys():
                    param_X_mean_var_dict[col_mean_name] = []
                    param_X_mean_var_dict[col_var_name] = []
                param_X_mean_var_dict[col_mean_name].append(param_X_time_df[x_col].mean())
                param_X_mean_var_dict[col_var_name].append(param_X_time_df[x_col].var())
            fpi_mean_name = 'front_fpi_mean'
            fpi_var_name = 'front_fpi_var'
            tpi_mean_name = 'front_tpi_mean'
            tpi_var_name = 'front_tpi_var'
            if fpi_mean_name not in param_X_mean_var_dict.keys():
                param_X_mean_var_dict[fpi_mean_name] = []
                param_X_mean_var_dict[fpi_var_name] = []
            if tpi_mean_name not in param_X_mean_var_dict.keys():
                param_X_mean_var_dict[tpi_mean_name] = []
                param_X_mean_var_dict[tpi_var_name] = []
            param_X_mean_var_dict[fpi_mean_name].append(cur_fpi_series.mean())
            param_X_mean_var_dict[fpi_var_name].append(cur_fpi_series.var())
            param_X_mean_var_dict[tpi_mean_name].append(cur_tpi_series.mean())
            param_X_mean_var_dict[tpi_var_name].append(cur_tpi_series.var())
            if ring_name not in param_y_mean_dict.keys():
                param_y_mean_dict[ring_name] = []
            param_y_mean_dict[ring_name].append(ring_num)
            if f_col not in param_y_mean_dict.keys():
                param_y_mean_dict[f_col] = []
            param_y_mean_dict[f_col].append(next_part_df[f_col].mean())
            if t_col not in param_y_mean_dict.keys():
                param_y_mean_dict[t_col] = []
            param_y_mean_dict[t_col].append(next_part_df[t_col].mean())
    final_pressure_X_time_df = None
    final_pressure_X_mean_var_df = None
    final_pressure_y_mean_df = None
    final_param_X_time_df = None
    final_param_X_mean_var_df = None
    final_param_y_mean_df = None
    if len(pressure_X_time_df_list) > 0:
        final_pressure_X_time_df = pd.concat(pressure_X_time_df_list, axis=0)
        final_pressure_X_mean_var_df = pd.DataFrame(pressure_X_mean_var_dict, index=None)
        final_pressure_y_mean_df = pd.DataFrame(pressure_y_mean_dict, index=None)
    if len(param_X_time_df_list) > 0:
        final_param_X_time_df = pd.concat(param_X_time_df_list, axis=0)
        final_param_X_mean_var_df = pd.DataFrame(param_X_mean_var_dict, index=None)
        final_param_y_mean_df = pd.DataFrame(param_y_mean_dict, index=None)
    result = {
        'pressure_X_time_df': final_pressure_X_time_df,
        'pressure_X_mean_var_df': final_pressure_X_mean_var_df,
        'pressure_y_mean_df': final_pressure_y_mean_df,
        'param_X_time_df': final_param_X_time_df,
        'param_X_mean_var_df': final_param_X_mean_var_df,
        'param_y_mean_df': final_param_y_mean_df
    }
    return result


def str_to_time(time_str):
    parts = time_str.split(' ')
    part1 = parts[0]
    part2 = parts[1]
    year, month, day = part1.split('/')
    hour, minute, second = part2.split(':')
    format_time = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    return format_time


# 时序数据的训练集和测试集划分
def time_series_train_test_split(X_file, y_file, x_cols, y_col):
    df_X = pd.read_csv(X_file, sep=',', index_col=None, header=0)
    df_y = pd.read_csv(y_file, sep=',', index_col=None, header=0)
    if IS_TEST:
        df_X = df_X[df_X['100001'] < 250]
        df_y = df_y[df_y['100001'] < 250]
    df_X = df_X[x_cols].astype('float32')
    df_y = df_y[y_col].astype('float32')

    sample_num = df_y.shape[0]
    input_dim = df_X.shape[1]

    x_scaler = StandardScaler()
    x_scaler.fit(df_X)
    x_array = x_scaler.transform(df_X)
    # print(x_array.shape)
    x_array = x_array.reshape((sample_num, TIME_LENGTH, input_dim))
    print(x_array.shape)

    y_scaler = StandardScaler()
    y_original_array = df_y.values.reshape(-1, 1)
    y_scaler.fit(y_original_array)
    y_array = y_scaler.transform(y_original_array)
    print(y_array.shape)

    train_X, test_X, train_y, test_y = train_test_split(x_array, y_array, test_size=TEST_SIZE, shuffle=True)
    return input_dim, x_scaler, y_scaler, train_X, test_X, train_y, test_y


if __name__ == '__main__':
    file_dir = 'data_ring_extracted_by_day'
    preprocessed_file_dir = 'preprocessed_data'
    is_fault = False
    data_construct(file_dir, preprocessed_file_dir, is_fault=is_fault)

