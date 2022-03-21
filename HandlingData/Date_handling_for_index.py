from dateutil import parser
import sys
from datetime import date
import datetime
import numpy as np

import pandas as pd

from HandlingData.stationarityTest import stationarity_testing, grangers_causation_matrix


def error_message():
    print("Kindly check the index for the date format")


def filling_method():
    print('How would you like to fill the missing data', '\n')
    print('Press 0  -----> Fill with Zero', '\n')
    print('Press 1  -----> Mean', '\n')
    print('Press 2  -----> Previous Value', '\n')
    choice = int(input("Enter the number: "))
    if choice < 0 or choice > 2:
        print("WARNING : Please check the number you have entered !!")
        filling_method()
    else:
        return choice


# If days are given we have to check for the missing days and add those days by either filling it with
# Mean , Mode , Previous Value
# We can fill the nan values in two either by MEAN or by Previous Values
def filling_nan_values(df):
    choice = filling_method()
    if choice == 0:
        df = df.fillna(0)
    if choice == 1:
        for i in range(len(df.columns)):
            col_name = df.columns[i]
            mean = df[col_name].mean()
            df[col_name] = df[col_name].fillna(mean)
        return df
    else:
        df = df.fillna(-sys.maxsize)
        for i in range(len(df.columns)):
            col_name = df.columns[i]
            for j in range(df.shape[0]):
                if j == 0 and df[col_name][j] == -sys.maxsize:
                    df[col_name][j] = df[col_name].mean()
                if df[col_name][j] == -sys.maxsize and j - 1 >= 0:
                    k = j - 1
                    df[col_name][j] = df[col_name][k]
        return df


def valide_date(df, first_date):
    sp_ch = ''
    date_format = ''
    today = date.today()
    current_date = today.strftime("%Y%m%d")
    if len(first_date) <= 10:
        if first_date.find('/') != -1:
            sp_ch = '/'
        elif first_date.find(' ') != -1:
            sp_ch = ' '
        elif first_date.find('-') != -1:
            sp_ch = '-'
        if len(first_date) == 4:
            date_format = '%Y'
        if 6 <= len(first_date) <= 7:
            date_format = '%Y' + sp_ch + '%m'
        if 7 < len(first_date) <= 10:
            date_format = '%Y' + sp_ch + '%m' + sp_ch + '%d'

    if len(first_date) > 10:
        date_format = '%Y-%m-%d' + ' ' + '%H:%M:%S'

    try:

        date_obj = bool(datetime.datetime.strptime(first_date, date_format))
        return True
    except ValueError:
        print("Incorrect data format, should be in the format of ", date_format)
        return error_message()


def eda_date(df):
    #df = df.set_index(df.columns[0])
    original_df_WDI = df.copy()
    flag = True
    df = df.set_index(df.columns[0])
    df = df.sort_index()
    first_date = str(df.index[0]).strip()
    char_sp = ''
    valid_date_flag = bool(valide_date(df, first_date))

    if valid_date_flag:

        if len(first_date) == 4 and flag:
            group_day = df.groupby(pd.PeriodIndex(data=df.index, freq='A'))
            results = group_day.sum()
            # fill the values with max size
            idx = pd.period_range(start=pd.Period(df.index[0], freq='A'), end=pd.Period(df.index[-1], freq='A'))
            df = results.reindex(idx, fill_value=None)
            flag = False
        if len(first_date) == 7 and flag:
            if first_date.find('/') != -1:
                char_sp = '/'
            elif first_date.find(' ') != -1:
                char_sp = ' '
            elif first_date.find('-') != -1:
                char_sp = '-'
                df['Date']=df.index
                for d in range(df.shape[0]):
                    date_s = df.index[d].strip()
                    df['Date'][d] = date_s
            df=df.set_index('Date')
            year = []
            month = []
            l1 = df.index
            for item in l1:
                if len(item.split(char_sp)) == 3:
                    item1, item2, f = item.split(char_sp)
                if len(item.split(char_sp)) == 2:
                    item1, item2 = item.split(char_sp)
                year.append(int(item1))
                month.append(int(item2))
            Date = []
            for y, m in zip(year, month):
                Date.append(date(y, m, 1))
            df.index = pd.to_datetime(Date)
            before_df = df.copy()
            group_day = df.groupby(pd.Grouper(freq='MS'))
            df = group_day.sum()
            # # fill the values with max size
            # idx = pd.period_range(start=pd.Period(df.index[0], freq='M'), end=pd.Period(df.index[-1], freq='M'), freq='M')
            df = before_df.reindex(df.index, fill_value=None)
            flag = False
        if 7 < len(str(df.index[0])) <= 10 and flag:
            group_day = df.groupby(pd.PeriodIndex(data=df.index, freq='D'))
            results = group_day.sum()
            # fill the values with max size
            idx = pd.period_range(min(df.index), max(df.index))
            df = results.reindex(idx, fill_value=None)
            flag = False
        if len(str(df.index[0])) > 10 and flag:
            df.index = pd.to_datetime(df.index)
            millis_arr = []
            date_format = '%Y-%m-%d %H:%M:%S'
            for i in range(df.shape[0]):
                date_obj = df.index[i]
                date_obj = datetime.datetime.strptime(str(date_obj), date_format)
                milli_seconds = date_obj.timestamp()
                millis_arr.append(int(milli_seconds))
            df_millis = pd.DataFrame(millis_arr)
            df_millis['lag'] = df_millis[df_millis.columns[0]].diff().dropna()
            mode_millis = df_millis.lag.mode(dropna=True)
            time_diff = int(mode_millis.values[0])
            freq = ''
            if time_diff < 60:
                freq = str(int(time_diff)) + 'S'
            elif 60 <= time_diff < 3600:
                freq = str(int(time_diff / 60)) + 'min'
            elif 3600 <= time_diff < 86400:
                freq = str(int(time_diff / 3600)) + 'H'
            elif time_diff >= 86400:
                freq = str(int(time_diff / 86400)) + 'D'
            original_df_WDI = original_df_WDI.set_index(original_df_WDI.columns[0]).asfreq(freq)
            df = df.reindex(original_df_WDI.index, fill_value=None)
        df = df.select_dtypes(include=np.number)
        df = filling_nan_values(df)
        # With this method we are going to the check the statioarity of the timeseries data
        # the diff df is the one in which we gonna perform the differencing
        if len(df.columns) > 1:
            df = grangers_causation_matrix(df, df.columns)
        original_df = df.copy()
        df_test = df[-10:]
        df = df[0:-10]
        diff_df = df.copy()
        combined_df = df.copy()
        diff_count = 0
        df = df.round(decimals=7)
        stationarity_testing(original_df, df, diff_df, combined_df, df_test, diff_count)
    else:
        print("-" * 50)
