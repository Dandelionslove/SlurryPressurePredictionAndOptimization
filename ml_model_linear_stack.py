from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from mlxtend.regressor import StackingCVRegressor
from functions import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate, cross_val_score
import joblib


def cal_mape(test_y, pred_y):
    return np.mean(np.abs((test_y - pred_y) / test_y)) * 100.0


def model_test(test_y, pred_y):
    mse = mean_squared_error(test_y, pred_y)
    mae = mean_absolute_error(test_y, pred_y)
    mape = cal_mape(test_y, pred_y)
    r2 = r2_score(test_y, pred_y)
    test_result = {
        'mse': mse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }
    return test_result


def lr_model_cv_selection(train_X, train_y, mode=0):
    model = LinearRegression()
    best_cv = 0
    best_r2 = 0.0
    model_name = 'lr{}'.format(mode)
    result_file = os.path.join(sub_gscv_result_dir, '{}.txt'.format(model_name))
    for cv in range(2, 11):
        scores = cross_validate(model, train_X, train_y, cv=cv,
                                scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], )
        val_r2 = np.mean(scores['test_r2'])
        print('cv = ', cv)
        print('val_r2:\t', val_r2)
        if best_cv == 0:
            best_cv = cv
            best_r2 = val_r2
            write_mode = 'w'
        else:
            write_mode = 'a'
            if val_r2 > best_r2:
                best_cv = cv
                best_r2 = val_r2
        with open(result_file, write_mode) as f:
            f.write('cv = {}, val_r2 = {}\n'.format(cv, val_r2))
            f.close()
    print('best cv:', best_cv)
    print('best r2:', best_r2)
    return best_cv


def construct_stacking_model():
    ridge = RidgeCV()
    adaboost_model = AdaBoostRegressor(learning_rate=0.1, n_estimators=450, loss='linear')
    rf_model = RandomForestRegressor(n_estimators=500, max_depth=35)
    xgb_model = XGBRegressor(n_estimators=450, learning_rate=0.1, max_depth=30)
    stack = StackingCVRegressor(regressors=(adaboost_model, rf_model, xgb_model), meta_regressor=ridge, cv=KFOLD_NUM)
    return stack


def model_train_test(cv=5, mode=0, is_save_model=True, model_type='lr'):
    y_scaler, train_X, train_y, test_X, test_y = data_preparation(is_time_series=False, mode=mode)
    train_y = train_y.ravel()
    test_y = test_y.ravel()

    if model_type is 'lr':
        model = LinearRegression()
    else:  # 'stack'
        model = construct_stacking_model()
    model_name = '{}{}'.format(model_type, mode)
    scores = cross_validate(model, train_X, train_y, cv=cv,
                            scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                            return_train_score=True, return_estimator=True, verbose=1)
    model = scores['estimator'][-1]
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    test_result = model_test(test_y, pred_y)
    test_y_org = y_scaler.inverse_transform(test_y)
    pred_y_org = y_scaler.inverse_transform(pred_y)
    print('model test')
    # 存储测试结果
    test_result_file = os.path.join(sub_test_result_dir, '{}.txt'.format(model_name))
    print_write_infos(test_result_file, **test_result)
    # 画拟合图
    plot_fitting_result(test_y_org, pred_y_org, model_name)
    # 将数据进行存储
    df = pd.DataFrame({
        'test_y': list(test_y_org),
        'pred_y': list(pred_y_org)
    }, index=None)
    data_file = os.path.join(sub_test_result_dir, 'pred_test_{}.csv'.format(model_name))
    df.to_csv(data_file, index=None)
    print('test data saved successfully!')
    # 将模型进行存储
    if is_save_model:
        model_file_name = '{}.m'.format(model_name)
        model_file = os.path.join(sub_model_dir, model_file_name)
        joblib.dump(model, model_file)


if __name__ == '__main__':
    mode = 0
    model_train_test(cv=KFOLD_NUM, mode=mode, is_save_model=True, model_type='lr')
    model_train_test(cv=KFOLD_NUM, mode=mode, is_save_model=True, model_type='stack')

