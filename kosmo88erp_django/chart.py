from __future__ import division
from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import warnings
warnings.filterwarnings("ignore")

import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

#import Keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split
# Import statsmodels.formula.api
import statsmodels.formula.api as smf

from sklearn.preprocessing import MinMaxScaler

import cx_Oracle as co

class Figure():
    #  DB Connecion
    dsn_tns = co.makedsn("kosmo88erp3.crn8w5ugntgi.ap-northeast-2.rds.amazonaws.com", "1521", service_name="ORCL")
    conn = co.connect(user="admin", password="tiger88!", dsn=dsn_tns)

    #initiate plotly
    pyoff.init_notebook_mode()

    #read the data in csv
    df_sales = pd.read_csv('../train.csv')

    #convert date field from string to datetime
    df_sales['date'] = pd.to_datetime(df_sales['date'], errors='coerce')
    # df_sales['date'] = df_sales['date'].dt.year.astype('str') + '-' + df_sales['date'].dt.month.astype('str') + df_sales['date'].dt.day.astype('str')
    # df_sales['date'] = pd.to_datetime(df_sales['date'])
    df_sales = df_sales.groupby('date').sales.sum().reset_index()

    df_diff_train = df_sales.copy()
    df_diff_train['prev_sales'] = df_diff_train['sales'].shift(1)
    df_diff_train = df_diff_train.dropna()
    df_diff_train['diff'] = (df_diff_train['sales'] - df_diff_train['prev_sales'])

    #create dataframe for transformation from time series to supervised
    df_supervisec_train = df_diff_train.drop(['prev_sales'], axis=1)
    #adding lags
    field_name_arr = []
    for inc in range(1,13):
        field_name = 'lag_' + str(inc)
        field_name_arr.append(field_name)
        df_supervisec_train[field_name] = df_supervisec_train['diff'].shift(inc)
        
    col = "diff ~ " + '+'.join(str(e) for e in field_name_arr)

    #drop null values
    df_supervisec_train = df_supervisec_train.dropna().reset_index(drop=True)



    # Define the regression formula
    # model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12', data=df_supervised)
    model_train = smf.ols(formula=col, data=df_supervisec_train)

    # Fit the regression
    model_fit_train = model_train.fit()

    # Extract the adjusted r-squared
    # regression_adj_rsq = model_fit.rsquared_adj
    regression_adj_rsq_train = model_fit_train.rsquared_adj

    df_model = df_supervisec_train.drop(['sales','date'],axis=1)

    ##-- SQL query
    query = """
    SELECT to_char(sl.update_date, 'YYYY-MM-DD') AS "date",
        SUM(supply_amount + tax_amount) AS "sales"
    FROM sales_slip sa, slip sl
    WHERE sa.slip_id = sl.id
    AND sl.state = 'Y'
    GROUP BY to_char(sl.update_date, 'YYYY-MM-DD')
    ORDER BY "date"
                """
    # Get a dataframe
    sales_slip = pd.read_sql(query, conn)

    #convert date field from string to datetime
    sales_slip['date'] = pd.to_datetime(sales_slip['date'], errors='coerce')
    sales_slip = sales_slip.groupby('date').sales.sum().reset_index()

    #차이를 모델링하기 위해 새 데이터 프레임을 만듭니다.
    df_diff = sales_slip.copy()
    # 이전 판매를 다음 행에 추가 
    df_diff['prev_sales'] = df_diff['sales'].shift(1)

    # null 값을 삭제하고 차이를 계산합니다. 
    df_diff = df_diff.dropna() 
    df_diff['diff'] = (df_diff['sales'] - df_diff['prev_sales'])

    #create dataframe for transformation from time series to supervised
    df_supervised = df_diff.drop(['prev_sales'], axis=1)

    #adding lags
    field_name_arr = []
    for inc in range(1,13):
        field_name = 'lag_' + str(inc)
        field_name_arr.append(field_name)
        df_supervised[field_name] = df_supervised['diff'].shift(inc)
        
    col = "diff ~ " + '+'.join(str(e) for e in field_name_arr)

    #drop null values
    df_supervised = df_supervised.dropna().reset_index(drop=True)

    #import MinMaxScaler and create a new dataframe for LSTM model
    
    df_model = df_supervisec_train.drop(['sales','date'],axis=1)
    df_test_model = df_supervised.drop(['sales','date'],axis=1)

    #split train and test set
    train_set = df_model[0:].values
    test_set =  df_test_model[0:].values

    #apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)

    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)

    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)

    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    model.fit(X_train, y_train, epochs=100, batch_size=30, verbose=1, shuffle=False)
    y_pred = model.predict(X_test,batch_size=32)

    #for multistep prediction, you need to replace X_test values with the predictions coming from t-1
    #reshape y_pred
    y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
    #rebuild test set for inverse transform
    pred_test_set = []
    for index in range(0,len(y_pred)):
        print(np.concatenate([y_pred[index],X_test[index]],axis=1))
        pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))

    #reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])

    #inverse transform
    pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

    #create dataframe that shows the predicted sales
    result_list = []
    sales_dates = list(df_sales[-7:].date)
    act_sales = list(df_sales[-7:].sales)
    for index in range(0,len(pred_test_set_inverted)):
        result_dict = {}
        result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
        result_dict['date'] = sales_dates[index+1]
        result_list.append(result_dict)
        
    df_result = pd.DataFrame(result_list)

    #for multistep prediction, replace act_sales with the predicted sales
    #merge with actual sales dataframe
    df_sales_pred = pd.merge(sales_slip,df_result,on='date',how='left')

    #plot actual and predicted
    plot_data = [
        go.Scatter(
            x=df_sales_pred['date'],
            y=df_sales_pred['sales'],
            name='actual'
        ),
        go.Scatter(
            x=df_sales_pred['date'],
            y=df_sales_pred['pred_value'],
            name='predicted'
        )
    ]
    plot_layout = go.Layout(
        title='Sales Prediction'
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)

    def __init__(self) -> None:
        pass
    
    def getFig(self):
        return self.fig