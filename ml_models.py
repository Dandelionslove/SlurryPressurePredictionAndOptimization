# coding:utf-8

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from functions import *
import matplotlib
import keras


# matplotlib.use('Qt5Agg')


# name_prefix = 't{}_s{}_k{}'.format(TIME_LENGTH, STEP, CV)


class MLModel:
    def __init__(self, model_type='rf', mode=0):
        '''
        :param model_type: 模型搭建方法
        :param mode: 模型类型，0--非断层，1--断层，2--非断层和断层混合
        '''
        self.model = None
        self.model_type = model_type
        self.model_name = '{}{}'.format(model_type, mode)

    def grid_search_cv_train(self, train_X, train_y, params, cv, result_file, write_mode, is_save_model=True):
        # 决策树
        if self.model_type == 'dt':
            model = DecisionTreeRegressor()
        # 集成算法
        elif self.model_type == 'rf':
            model = RandomForestRegressor()
        # elif self.model_type == 'adaboost':
        #     model = AdaBoostRegressor()
        elif self.model_type == 'xgb':
            model = xgb.XGBRegressor()
        elif self.model_type == 'gbdt':
            model = GradientBoostingRegressor()
        elif self.model_type == 'lgb':
            model = lgb.LGBMRegressor()
        else:  # self.model_type == 'adaboost':
            model = AdaBoostRegressor()
        grid_search = GridSearchCV(model, params, cv=cv, scoring='neg_mean_squared_error', verbose=2)
        grid_search_result = grid_search.fit(train_X, train_y)
        # gcv_res = grid_search_result.cv_results_
        best_params = grid_search_result.best_params_
        best_score = np.abs(grid_search_result.best_score_)
        gscv_result = {
            'cv': cv,
            'best_params': best_params,
            'best_score': best_score
        }
        print_write_infos(result_file, write_mode, **gscv_result)
        self.model = grid_search_result.best_estimator_
        if is_save_model:
            self.model.fit(train_X, train_y)
            model_file_name = self.model_name + '.m'
            model_file = os.path.join(sub_model_dir, model_file_name)
            joblib.dump(self.model, model_file)
        return best_params, best_score

    def train(self, train_X, train_y, cv, **kwargs):
        if self.model == 'dt':
            model = DecisionTreeRegressor(max_depth=kwargs['max_depth'], max_features=kwargs['max_features'])
        elif self.model == 'rf':
            model = RandomForestRegressor(n_estimators=kwargs['n_estimators'], max_depth=kwargs['max_depth'],
                                          max_features=kwargs['max_features'])
        elif self.model == 'xgb':
            model = xgb.XGBRegressor(max_depth=kwargs['max_depth'], n_estimators=kwargs['n_estimators'],
                                     objective=kwargs['objective'], learning_rate=kwargs['learning_rate'],
                                     reg_lambda=kwargs['lambda'], reg_alpha=kwargs['alpha'])
        elif self.model == 'gbdt':
            model = GradientBoostingRegressor(loss=kwargs['loss'], learning_rate=kwargs['learning_rate'],
                                              n_estimators=kwargs['n_estimators'], criterion=kwargs['criterion'],
                                              max_depth=kwargs['max_depth'], max_features=kwargs['max_features'])
        else:  # lgb
            model = lgb.LGBMRegressor(max_depth=kwargs['max_depth'], learning_rate=kwargs['learning_rate'])
        scores = cross_validate(model, train_X, train_y, cv=cv,
                                scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                                return_train_score=True)
        cross_train_r2 = np.mean(scores['train_r2'])
        cross_train_mse = np.mean(np.abs(scores['train_neg_mean_squared_error']))
        cross_train_mae = np.mean(np.abs(scores['train_neg_mean_absolute_error']))

        cross_val_r2 = np.mean(scores['test_r2'])
        cross_val_mse = np.mean(np.abs(scores['test_neg_mean_squared_error']))
        cross_val_mae = np.mean(np.abs(scores['test_neg_mean_absolute_error']))
        result = {
            'cross_train_r2': cross_train_r2,
            'cross_train_mse': cross_train_mse,
            'cross_train_mae': cross_train_mae,
            'cross_val_r2': cross_val_r2,
            'cross_val_mse': cross_val_mse,
            'cross_val_mae': cross_val_mae
        }

        result_file = os.path.join(sub_test_result_dir, 'cv{}_{}.txt'.format(cv, self.model_name))
        print_write_infos(result_file, 'w', **result)
        model.fit(train_X, train_y)
        self.model = model

    def load_model(self, model_file):
        self.model = joblib.load(model_file)

    def test(self, test_X, test_y):
        if self.model is None:
            print('model is not trained')
            return
        pred_y = self.model.predict(test_X)
        r2 = r2_score(test_y, pred_y)
        mse = mean_squared_error(test_y, pred_y)
        mae = mean_absolute_error(test_y, pred_y)
        mape = np.mean(np.abs((test_y - pred_y) / test_y)) * 100

        test_result = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'mape': mape
        }
        result_file = os.path.join(sub_test_result_dir, '{}.txt'.format(self.model_name))
        print_write_infos(result_file, **test_result)

        return pred_y, r2, mse, mae, mape

    def draw_pred_result(self, test_y_org, pred_y_org):
        plot_fitting_result(test_y_org, pred_y_org, self.model_name)


# 基于不同的cv进行模型选择
def grid_search_cv_model_selection_with_different_cv(train_X, train_y, params, model_type='rf', mode=0):
    # 网格搜索获取最优超参数
    model_construction = MLModel(model_type=model_type, mode=mode)
    result_file_name = model_construction.model_name + '.txt'
    result_file = os.path.join(sub_gscv_result_dir, result_file_name)

    final_best_cv = 0
    final_best_score = None
    for cv in range(MIN_CV, MAX_CV + 1):
        print('\033[0;36mcv=%d\033[0m' % cv)
        if cv == MIN_CV:
            write_mode = 'w'  # 'w'
        else:
            write_mode = 'a'
        best_params, best_score = model_construction.grid_search_cv_train(train_X, train_y, params, cv, result_file,
                                                                          write_mode)
        if final_best_cv == 0:
            final_best_cv = cv
            final_best_score = best_score
        else:
            if final_best_score > best_score:
                final_best_score = best_score
                final_best_cv = cv
    with open(result_file, 'a') as f:
        print('*' * 12)
        f.write('\n' + ('*' * 12))
        if final_best_cv is not None:
            f.write('\nbest cv:\t%d\n' % final_best_cv)
            print('final best cv:\t%d' % final_best_cv)
        f.close()


def grid_search_cv_model_selection(train_X, train_y, params, model_type='rf', mode=0):
    # 网格搜索获取最优超参数
    model_construction = MLModel(model_type=model_type, mode=mode)
    result_file_name = model_construction.model_name + '.txt'
    result_file = os.path.join(sub_gscv_result_dir, result_file_name)

    best_params, best_score = model_construction.grid_search_cv_train(train_X,
                                                                      train_y,
                                                                      params,
                                                                      KFOLD_NUM,
                                                                      result_file,
                                                                      'w')


def test_model(y_scaler, test_X, test_y, model_type='rf', mode=0):
    model = MLModel(model_type=model_type, mode=mode)
    model_file = os.path.join(sub_model_dir, '{}.m'.format(model.model_name))
    model.load_model(model_file)
    pred_y, test_r2, test_mse, test_mae, test_mape = model.test(test_X, test_y)
    # 数据还原
    test_y_org = y_scaler.inverse_transform(test_y.reshape(-1, 1)).reshape(-1)
    pred_y_org = y_scaler.inverse_transform(pred_y.reshape(-1, 1)).reshape(-1)
    # 将预测的数据进行保存，以便后续画图和分析
    pred_test_result_file = os.path.join(sub_test_result_dir, 'pred_test_%s.csv' % model.model_name)
    pred_test_df = pd.DataFrame({
        'test_y': test_y_org,
        'pred_y': pred_y_org
    })
    pred_test_df.to_csv(pred_test_result_file, index=None)
    model.draw_pred_result(test_y_org, pred_y_org)


def pressure_prediction(model_type='rf', mode=0, is_grid=True):
    '''
        泥水仓压力预测模型
        @:param is_grid: 是否使用网格搜索，True/False
        @:param mode: 训练数据模式，0--非断层数据，1--断层数据，2--断层和非断层一起的数据
    '''
    print('\033[1;35m {} \033[0m'.format(model_type))
    # 数据准备
    y_scaler, train_X, train_y, test_X, test_y = data_preparation(is_time_series=False, mode=mode)

    if is_grid:
        # 网格超参数搜索
        if model_type == 'dt':
            grid_params = {
                'max_depth': range(20, 41, 5)
            }
        elif model_type == 'rf':
            grid_params = {
                'n_estimators': range(100, 501, 10),
                'max_depth': range(20, 41, 5)
            }
        elif model_type == 'xgb':
            grid_params = {
                'n_estimators': range(100, 501, 10),
                'max_depth': range(20, 41, 5),
                'objective': ['reg:squarederror'],
                'learning_rate': np.arange(0.01, 0.3, 0.01),
            }
        elif model_type == 'gbdt':
            grid_params = {
                'n_estimators': range(100, 501, 10),
                'max_depth': range(20, 41, 5),
                'learning_rate': np.arange(0.01, 0.3, 0.01),
                'loss': ['ls', 'lad'],
                'criterion': ['mse'],
            }
        elif model_type == 'lgb':  # lgb
            grid_params = {
                'n_estimators': range(100, 501, 10),
                'max_depth': range(20, 41, 5),
                'learning_rate': np.arange(0.01, 0.3, 0.01),
                'boosting_type': ['goss', 'gbdt', 'rf', 'dart']
            }
        else:  # adaboost
            grid_params = {
                'learning_rate': np.arange(0.01, 0.3, 0.01),
                'n_estimators': range(100, 501, 10),
                'loss': ['linear', 'square', 'exponential'],
                'base_estimator': [DecisionTreeRegressor(max_depth=30)]
            }

        grid_search_cv_model_selection(train_X, train_y, grid_params, model_type=model_type, mode=mode)
    else:
        # 测试最优模型
        test_model(y_scaler, test_X, test_y, model_type=model_type, mode=mode)


if __name__ == '__main__':
    # ['dt', 'rf', 'gbdt', 'xgb', 'lgb']
    model_type_list = ['dt', 'rf', 'gbdt', 'xgb', 'lgb', 'adaboost']
    # 0--非断层，1--断层，2--断层+非断层
    mode = 0
    # 网格超参数搜索
    for model_type in model_type_list:
        pressure_prediction(model_type=model_type, mode=mode, is_grid=True)
        # 评估模型
        pressure_prediction(model_type=model_type, mode=mode, is_grid=False)

