import statsmodels
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from ForecastingModel.multivariate import multivariate_models
from ForecastingModel.univariate import univariate_models
import pandas as pd
import numpy as np


# To check whether the given series is Univariant or Multivariant
# This methos return
# True -- if it is Univariate
# False --if it is Multivariate
def type_of_variate(df):
    column = len(df.columns)
    if column > 1:
        return False
    else:
        return True


def adf_test(series):
    result = adfuller(series, autolag='AIC')
    if result[1] <= 0.05:
        return True
    else:
        return False


def differencing_series(df, combined_df, diff_count):
    for col in df.columns:
        col_name = col + '_' + diff_count
        combined_df[col_name] = df[col].diff().dropna()
    combined_df.dropna()
    df = df.diff().dropna()
    return df, combined_df


def stationary_result(df):
    stationary = []
    for i in range(len(df.columns)):
        stationary.append(adf_test(df[df.columns[i]]))
    return stationary


# df- The original data frame
# diff_df- (Initially it is a copy of df)The data frame in which we gonna perform the differencing
# diff_count Is to measure the differencing count
# combined _df  contains both the df and all the diff df
# stationarity_testing(df, diff_df, combined_df,df_test,  diff_count, 0, 0)
def stationarity_testing(original_df, df, diff_df, combined_df, df_test, diff_count, flag=0, true_count=0, max_diff=5):
    stationary = stationary_result(diff_df)
    # print('diff Count', diff_count)
    for r in range(len(stationary)):
        if stationary[r]:
            true_count += 1
        if not stationary[r] and flag == 0 and diff_count <= max_diff:
            diff_count += 1
            diff_df, combined_df = differencing_series(diff_df, combined_df, str(diff_count))
            flag += 1
        # When diff count reaches the max limit 5 we remove the particular column in all the data frames from diff_df
        # and from combined_df
        if not stationary[r] and flag == 0 and diff_count > max_diff:
            diff_df.drop(columns=[diff_df.columns[r]], axis=1, inplace=True)
            n = diff_count
            combined_df.drop(columns=[combined_df.columns[r]], axis=1, inplace=True)
            while n > 0:
                col = df.columns[r] + '_' + str(n)
                combined_df.drop(columns=[combined_df.columns[col]], axis=1, inplace=True)
                n -= 1

    if true_count != len(stationary):
        stationarity_testing(original_df, df, diff_df, combined_df, df_test, diff_count, 0, 0)
    if flag == 0 and true_count == len(stationary):
        # Checking whether the df is univariate or multivariate
        variate = type_of_variate(df)
        if variate:
            return univariate_models(original_df, df, diff_df, combined_df, diff_count, df_test)
        else:
            return multivariate_models(df, diff_df, combined_df, diff_count, df_test)


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            try:

                test_result = grangercausalitytests(data[[r, c]], maxlag=12, verbose=False)
                p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(12)]
                if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
            except statsmodels.tools.sm_exceptions.InfeasibleTestError:
                data.drop([r], axis=1, inplace=True)
                return grangers_causation_matrix(data, data.columns)

    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    bool_df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for row in range(len(df.index)):
        for col in range(len(variables)):
            if df[df.columns[col]][row] <= 0.05:
                bool_df[bool_df.columns[col]][row] = 0  # True
            else:
                bool_df[bool_df.columns[col]][row] = 1  # False
    bool_count = {}
    final_count = {}
    for row in range(len(variables)):
        count = 0
        for col in range(len(variables)):
            if bool_df[bool_df.columns[col]][row] == 1:
                count += 1
        bool_count[row] = count
        final_count = sorted(bool_count.items(), key=
        lambda kv: (kv[1], kv[0]))
    ar = list()
    for i in final_count:
        ar.append(i[0])
    final_op = data.copy()
    for i in range(12, len(ar)):
        final_op.drop([data.columns[ar[i]]], axis=1, inplace=True)
    return final_op
