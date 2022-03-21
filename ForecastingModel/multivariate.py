import numpy as np
from Cython import inline
from pmdarima.compat import matplotlib
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import grangercausalitytests


def message(col):
    return (print('Kindly verify the column ', col, ' That column residual contains  correaltion pattern'))


# df- The original data frame
# diff_df-The data frame in which we gonna perform the differencing
# diff_count  the differencing lag count

def vector_auto_regression(df, diff_df, diff_count):
    model = VAR(diff_df)
    x = model.select_order(maxlags=12)
    model_fitted = model.fit(x.aic)
    out = durbin_watson(model_fitted.resid)
    for col, val in zip(df.columns, out):
        # print(col, ' : ', round(val, 2))
        if 1 > val >= 0:
            return message(col)
        elif 3 < val <= 4:
            return message(col)
    lag_order = model_fitted.k_ar
    forecast_input = diff_df.values[-lag_order:]
    fc = model_fitted.forecast(y=forecast_input, steps=10)
    df_dummy = df.copy()
    df_f = pd.DataFrame(fc)

    for col in range(len(df.columns)):
        for i in range(len(df_f[col])):
            df_dummy[df_dummy.columns[col]][i] = df_f[df_f.columns[col]][i]
        name = df.columns[col]
        if diff_count > 0:
            rename = df.columns[col] + '_' + str(diff_count)
        else:
            rename = df.columns[col] + '_' + str('forecast')
        df_dummy.rename(columns={name: rename}, inplace=True)
    return df_dummy


def plotting_forecasted_values(df_test, forecasted_df):
    print('Actual Value')
    print(df_test)
    fig, axis = plt.subplots(nrows=int(len(df_test.columns) / 2), ncols=2, dpi=150, figsize=(8, 8))
    for i, (col, ax) in enumerate(zip(df_test.columns, axis.flatten())):
        forecasted_df[col + '_forecast'].plot(legend=True, ax=ax).autoscale(axis='x', tight=True)
        df_test[col].plot(legend=True, ax=ax)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
    plt.show()
    plt.tight_layout();


def multivariate_models(df, diff_df, combined_df, diff_count, df_test):
    op = df.copy()
    out = coint_johansen(diff_df, -1, diff_count)
    alpha = 0.05
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1 - alpha)]]

    def adjust(val, length=6):
        return str(val).ljust(length)

    # print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--' * 20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        pass
        # print(adjust(col), ':: ', adjust(round(trace, 2), 9), ">", adjust(cvt, 8), ' =>  ', trace > cvt)
    forecasted_df = vector_auto_regression(df_test, diff_df, diff_count)
    if diff_count > 0:
        forecasted_df = reinvert_transformation(df, diff_df, diff_count, forecasted_df)
    print("The Forecasted value for next 10 steps :")
    if forecasted_df.iloc[:, len(df_test.columns):].empty:
        print(forecasted_df)
    else:
        print(forecasted_df.iloc[:, len(df_test.columns):])

    plotting_forecasted_values(df_test, forecasted_df)


# For each lag from ascending order we have to subrateact from the higher order lag one by one
# for eg lets take if the diff_count =3
# value= lastvalues of (lag(1)-lag(2)-lag(3))
# from which we will get values and we have to cumsum with the forecasted valu
def each_lag(diff_count, df):
    arr = []
    for i in range(1, diff_count):
        arr[i - 1] = df.iloc[-i]
    value = arr[0]
    for i in range(1, diff_count):
        value = value - arr[i]
    return value


# Reversing the differenced variables back to inital stage with the help of differnced count
def reinvert_transformation(df, diff_df, diff_count, forecasted_df):
    while diff_count > 1:
        for col in df.columns:
            forecasted_df[str(col) + '_' + str(diff_count - 1)] = each_lag(diff_count, df[col]) + \
                                                                  forecasted_df[
                                                                      str(col) + '_' + str(diff_count)].cumsum()
    if diff_count == 1:
        for col in df.columns:
            forecasted_df[str(col) + '_forecast'] = df[col].iloc[-1] + forecasted_df[str(col) + '_' + str(1)].cumsum()
    if diff_count == 0:
        for col in df.columns:
            forecasted_df[str(col) + '_forecast'] = forecasted_df[str(col) + '_' + str(0)].cumsum()
    return forecasted_df
