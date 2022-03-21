import sys

import pandas as pd
import numpy as np
from datetime import date

from HandlingData.stationarityTest import stationarity_testing


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
            df[col_name] = df[col_name].fillna(df[col_name].mean())
        return df
    else:
        df = df.fillna(-sys.maxsize)
        for i in range(len(df.columns)):
            col_name = df.columns[i]
            for j in range(len(df[df.columns[0]])):
                if j == 0 and df[col_name][j] == -sys.maxsize:
                    df[col_name][j] = df[col_name].mean()
                if df[col_name][j] == -sys.maxsize and j - 1 >= 0:
                    k = j - 1
                    df[col_name][j] = df[col_name][k]
        return df


def date_Handling(df):
    column_name = df.columns
    flag = True
    column_name_arr = np.array(column_name.values)
    for col in column_name_arr:
        # If Year and month both are given in a seperated way , We have to combine the year and month
        if (col == 'Year' or col == 'year') and flag:
            for month in column_name_arr:
                if month == 'Month' or col == 'month':
                    for day in column_name_arr:
                        if day == 'Day' or day == 'day':
                            Date = []
                            for y, m, d in zip(df[col], df[month], df[day]):
                                Date.append(date(y, m, 1))
                            df['Date'] = Date
                            df.drop(columns=[df[day]], axis=1, inplace=True)
                            flag = False
                    if flag:
                        Date = []
                        for y, m in zip(df[col], df[month]):
                            Date.append(date(y, m, 1))
                        df['Date'] = Date
                    df.index = pd.to_datetime(df.Date)
                    df.drop(columns=[df[month], df[col], df['Date']], axis=1, inplace=True)
                    flag = False
            if flag:
                df.index = pd.to_datetime(df[col])
                df.drop(columns=[df[col]], axis=1, inplace=True)
                flag = False

        if (col == 'Date' or col == 'date') and flag:
            range_of_date = pd.DataFrame(pd.date_range(df[col].min(), df[col].max()))
            range_of_date['Date'] = range_of_date[range_of_date.columns]
            range_of_date.drop(columns=[range_of_date.columns[0]], axis=1, inplace=True)
            if len(range_of_date.Date) > len(df[col]):
                group_day = df.groupby(pd.PeriodIndex(data=df.Date, freq='D'))
                results = group_day.sum()
                # fill the values with max size
                idx = pd.period_range(min(df.Date), max(df.Date))
                df = results.reindex(idx, fill_value=None)
                flag = False
        # If the year-month are given in the column of  month then this block is used for that
        if (col == 'Month' or col == 'month') and flag:
            df.index = pd.to_datetime(df[col])
            df.drop(columns=[col], axis=1, inplace=True)
            flag = False
    # Taking only the columns with the numeric values and neglecting the other columns
    df = df.select_dtypes(include=np.number)
    # This is method is to fill the nan values
    df = filling_nan_values(df)
    # With this method we are going to the check the statioarity of the timeseries data
    # the diff df is the one in which we gonna perform the differencing
    df_test=df[-10:]
    df= df[0:-10]
    diff_df = df.copy()
    combined_df =df.copy()
    diff_count = 0
    stationarity_testing(df, diff_df,combined_df,df_test, diff_count)
