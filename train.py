from statistics import mode
import optuna
import time
import joblib
import imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pkg_resources import parse_requirements
from datetime import date, datetime
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from cProfile import label
from cgi import test
from fileinput import filename
from sklearn.utils import shuffle
from subprocess import list2cmdline
from tkinter.tix import Tree
from turtle import penup
from unicodedata import name
# py train.py
def to_supervised_con(data, n_in=1, n_out=1):
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df['consumption'].shift(i))
        names += [('con(t-%d)' % (i))]
    
    for i in range(0, n_out):
        cols.append(df['consumption'].shift(-i))
        if i == 0:
            names += [('con(t)')]
        else:
            names += [('con(t+%d)' % (i))]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 刪除那些包含空值(NaN)的行
    agg.dropna(inplace=True)
    return agg    

def to_supervised_gen(data, n_in=1, n_out=1):
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df['generation'].shift(i))
        names += [('gen(t-%d)' % (i))]

    for i in range(0, n_out):
        cols.append(df['generation'].shift(-i))
        if i == 0:
            names += [('gen(t)')]
        else:
            names += [('gen(t+%d)' % (i))]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 刪除那些包含空值(NaN)的行
    agg.dropna(inplace=True)
    return agg
    
if __name__ == '__main__':
    """
    # 隨便看一下資料
    df = pd.DataFrame(columns=['generation', 'consumption', 'index'])
    for i in range(50):
        fileName = 'training_data\\target'+str(i)+'.csv'
        data = pd.read_csv(fileName)
        conSum = data['consumption'].sum()
        genSum = data['generation'].sum()
        df = df.append({'generation': genSum, 'consumption':conSum, 'index': i}, ignore_index=True)

    df = df.sort_values(['consumption', 'generation'], ignore_index=True)
    print(df)
    x = np.arange(50)
    plt.bar(x, df['consumption'], label='consumpotion Sum')
    plt.bar(x, df['generation'], label='generation Sum')
    plt.legend()
    plt.show()
    """
    target = [4, 37, 44, 3, 19, 10, 12, 31, 5, 27, 47, 42, 0, 35, 22, 43, 14, 25, 43]
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    wholeGenData = pd.DataFrame()
    wholeConData = pd.DataFrame()
    
    # Combine Target File Together.
    for i in target:
        fileName = './training_data/target'+str(i)+'.csv'
        data = pd.read_csv(fileName, parse_dates=['time'], index_col=0,date_parser=custom_date_parser, squeeze=True)
        genData = to_supervised_gen(data, n_in=23, n_out=25)
        conData = to_supervised_con(data, n_in=23, n_out=25)
        wholeGenData = wholeGenData.append(genData)
        wholeConData = wholeConData.append(conData)

    # Shuffle Data.
    wholeGenData = wholeGenData.sample(frac=1).reset_index(drop=True)
    wholeConData = wholeConData.sample(frac=1).reset_index(drop=True)
    
    # Normalization.
    scalar = MinMaxScaler(feature_range=(0, 1))
    wholeGenData = scalar.fit_transform(wholeGenData)
    joblib.dump(scalar, 'genScalar.save')

    scalar = MinMaxScaler(feature_range=(0, 1))
    wholeConData = scalar.fit_transform(wholeConData)
    joblib.dump(scalar, 'conScalar.save')

    # Train the Generation Model.
    Xtrain = wholeGenData[:-22000, :-24]    # 倒數22000資料作為測試資料:)
    Ytrain = wholeGenData[:-22000, -24:]
    Xtest = wholeGenData[-22000:, :-24]
    Ytest = wholeGenData[-22000:, -24:]

    model = XGBRegressor()
    model = model.fit(Xtrain, Ytrain)
    tmpStr = './genModel/'+time.strftime("%Y-%m-%d-%H-%M", time.localtime())+'-model.h5'
    model.save_model(tmpStr)
    score = model.score(Xtest, Ytest)
    print(score)
    # model.load_model('./genModel/2022-05-17-00-46-model.h5')
    # input = Xtest[-1, :]
    # input = input.reshape(1, 24)
    # y = model.predict(input)
    # print(y)
    # print(Ytest[-1, :])

    # Train the Consumption Model.
    Xtrain = wholeConData[:-22000, :-24]    # 倒數22000資料作為測試資料:)
    Ytrain = wholeConData[:-22000, -24:]
    Xtest = wholeConData[-22000:, :-24]
    Ytest = wholeConData[-22000:, -24:]

    model = XGBRegressor()
    model = model.fit(Xtrain, Ytrain)
    tmpStr = './conModel/'+time.strftime("%Y-%m-%d-%H-%M", time.localtime())+'-model.h5'
    model.save_model(tmpStr)
    score = model.score(Xtest, Ytest)
    print(score)

    """
    Xtrain = ShuffleData.iloc[:-22000, :-48]
    Ytrain = ShuffleData.iloc[:-22000, -48:]
    Xtest = wholeData.iloc[-22000:, :-48]
    Ytest = wholeData.iloc[-22000:, -48:]

    model = XGBRegressor()
    model = model.fit(Xtrain, Ytrain)

    tmp_str = './model/'+time.strftime("%Y-%m-%d-%H-%M", time.localtime())+'-model.h5'
    model.save_model(tmp_str)
    score = model.score(Xtest, Ytest)
    print(score)
    """
