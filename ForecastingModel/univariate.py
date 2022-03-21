import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.tsa.ar_model import ar_select_order, AutoReg
from matplotlib import pyplot as plt
from ForecastingModel.tools_for_modelselection import mean_sq_err


# df- The original data frame
# diff_df-The data frame in which we gonna perform the differencing
# diff_count  the differencing lag count
# train_X,test_X is the lagged values ie the predicted values
# train_Y,test_Y is the actual values
def naive_model(df, mse_predictions, model_train,size=10):
    lag_name = ''
    name = ''
    train, test = df[0:df.shape[0] - size], df[df.shape[0] - size:]
    column_name = np.array(df.columns.values)
    if len(column_name[0]) < len(column_name[1]):
        lag_name = column_name[1]
        name = column_name[0]
    else:
        lag_name = column_name[0]
        name = column_name[1]
    train_X, train_Y = train[lag_name], train[name]
    test_X, test_Y = test[lag_name], test[name]
    mse = mean_sq_err(test_Y, test_X)
    mse_predictions[mse] = test_X
    model_train[mse]=" Naive Model"




def auto_arima_model(train, test, mse_predictions,model_name):
    arima_model = auto_arima(train, start_p=0, d=1, start_q=0, max_p=40, max_d=5, max_q=40, error_action='warn',
                             start_P=0,
                             D=1, start_Q=0, max_P=5, max_D=5, max_Q=5, stepwise=True)
    prediction = pd.DataFrame(arima_model.predict(n_period=10), columns=['Prediction'])
    prediction['Date'] = test.index
    prediction.index = prediction['Date']
    prediction.drop(columns=['Date'], axis=1, inplace=True)

    mse = mean_sq_err(prediction, test[test.columns[0]])
    mse_predictions[mse] = prediction
    model_name[mse]="Auto Arima Model"


def auto_regression_model(train, test, size, mse_predictions, ar_lag,model_name):
    model = AutoReg(train, lags=ar_lag, old_names=False)
    model_fit = model.fit()
    prediction = model_fit.predict(start=len(train), end=len(train) + size - 1, dynamic=False)
    mse = mean_sq_err(test, prediction)
    mse_predictions[mse] = prediction
    model_name[mse]="Auto Regression"


def auto_regression_walk_forward(train, test, size, mse_predictions, ar_lag,model_name):
    data = train.copy()
    prediction = []
    for t in test[test.columns[0]]:
        model = AutoReg(data, ar_lag, old_names=False)
        model_fit = model.fit()
        y = model_fit.predict(start=len(data), end=len(data) + size)
        prediction.append(y.values[0])
        data = np.append(data, t)
        data = pd.Series(data)
    prediction = pd.DataFrame(prediction, index=test.index)
    prediction = prediction.rename(columns={prediction.columns[0]: test.columns[0]})
    mse = mean_sq_err(test, prediction)
    mse_predictions[mse] = prediction
    model_name[mse]= "AR Walk forward"


def predicted_output(mse_predictions, df_test,model_name):

    key_list = []
    for key, value in mse_predictions.items():
        key_list.append(key)
    key_list.sort()

    output = mse_predictions[key_list[0]]
    print('The Model Name')
    print(model_name[key_list[0]])
    print('The Forecasted Output : ')
    print(output)
    print('The MSE for the selected model is  :',key_list[0])
    output= pd.DataFrame(output,index=df_test.index)
    output['test']=df_test[df_test.columns[0]]
    plt.figure(figsize=(12, 5))
    plt.xlabel('Time index')
    # df_test[df_test.columns[0]]
    ax1 = output[output.columns[0]].plot(color='blue', grid=True, label='Actual')
    ax2 = output[output.columns[1]].plot(color='red', grid=True, secondary_y=True, label='Predicted')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    plt.legend(h1 + h2, l1 + l2, loc=2)
    plt.show()


def univariate_models(original_df, df, diff_df, combined_df, diff_count, df_test):
    model_name={}
    mse_predictions = {}
    size = 10
    naive_df = original_df.copy()
    lag_name = naive_df.columns[0] + '_lag'
    naive_df[lag_name] = naive_df[naive_df.columns[0]].shift(1)
    naive_df = naive_df.dropna()
    train, test = df.copy(), df_test.copy()
    naive_model(naive_df, mse_predictions,model_name)
    # ARIMA model p,d,q Auto selection model
    auto_arima_model(train, test, mse_predictions,model_name)
    # AutoRegression Model
    # selecting order of lag to check the correlation
    lag_order = ar_select_order(train, maxlag=int(0.2 * train.shape[0]), old_names=False, ic='aic', missing='drop')
    ar_lag = lag_order.ar_lags
    auto_regression_model(train, test, size, mse_predictions, ar_lag,model_name)
    # Walk Forward Auto Regression
    auto_regression_walk_forward(train, test, size, mse_predictions, ar_lag,model_name)
    print('The Actual Output of Test ')
    print(df_test)
    predicted_output(mse_predictions, df_test,model_name)
