import optuna
import time
import imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pkg_resources import parse_requirements
from datetime import date, datetime
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
def to_supervised(data, n_in=1, n_out=1):
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df['generation'].shift(i))
        names += [('gen(t-%d)' % (i))]
        cols.append(df['consumption'].shift(i))
        names += [('con(t-%d)' % (i))]
    
    for i in range(0, n_out):
        cols.append(df['generation'].shift(-i))
        if i == 0:
            names += [('gen(t)')]
        else:
            names += [('gen(t+%d)' % (i))]
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
    wholeData= pd.DataFrame()
    for i in target:
        fileName = './training_data/target'+str(i)+'.csv'
        data = pd.read_csv(fileName, parse_dates=['time'], index_col=0,date_parser=custom_date_parser, squeeze=True)
        Ndata = to_supervised(data, n_in=23, n_out=2)
        wholeData = wholeData.append(Ndata)
    # print(Ndata)
    ShuffleData = wholeData.sample(frac=1).reset_index(drop=True)

    Xtrain = ShuffleData.iloc[:-22000, :-2]
    Ytrain = ShuffleData.iloc[:-22000, -2:]
    Xtest = wholeData.iloc[-22000:, :-2]
    Ytest = wholeData.iloc[-22000:, -2:]

    model = XGBRegressor()
    model = model.fit(Xtrain, Ytrain)

    tmp_str = './model/'+time.strftime("%Y-%m-%d-%H-%M", time.localtime())+'-model.h5'
    model.save_model(tmp_str)
    score = model.score(Xtest, Ytest)
    print(score)
