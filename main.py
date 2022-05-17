import imp
from multiprocessing.dummy import JoinableQueue
from pyexpat import model
from statistics import mode
from time import time
import joblib
import pandas as pd
import numpy as np
import argparse
from pyparsing import condition_as_parse_action
from xgboost import XGBRegressor
from datetime import datetime, timedelta
# You should not modify this part.
def config():
   

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):
    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return

def to_supervised(data, n_in=1, n_out=1, var=''):
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df[var].shift(i))
        names += [('var(t-%d)' % (i))]

    for i in range(0, n_out):
        cols.append(df[var].shift(-i))
        if i == 0:
            names += [('var(t)')]
        else:
            names += [('var(t+%d)' % (i))]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 刪除那些包含空值(NaN)的行
    agg.dropna(inplace=True)
    return agg

if __name__ == "__main__":
    
    args = config()
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    conData = pd.read_csv(args.consumption, parse_dates=['time'], date_parser=custom_date_parser, index_col=0, squeeze=True)
    genData = pd.read_csv(args.generation, parse_dates=['time'], date_parser=custom_date_parser, index_col=0, squeeze=True)
    time = conData.index
    # print(Condata)

    # Predict Generation.
    # 1. 把模型與scalar載入
    scalar = joblib.load('genScalar.save') 
    model = XGBRegressor()
    model.load_model('./genModel/2022-05-17-01-15-model.h5')
    
    # 2. 把資料轉成model需求ㄉinput格式
    genData = to_supervised(genData, n_in=23, n_out=1, var='generation')
    genData = scalar.fit_transform(genData)
    X = genData[0, :] # 測試看看準不準==
    # X = genData[-1, :]
    X = X.reshape(1, 24)  

    # 3. 預測yo
    genPredict = model.predict(X)
    invGen = scalar.inverse_transform(genPredict)
    invGen = invGen.reshape(24,)
    # roundGen = [round(num, 2) for num in invGen]
    # print('Generation: ')
    # print(roundGen)
    # print('--------------------------------------')

    # Predict Consumption.
    # 1. 把模型與scalar載入
    scalar = joblib.load('conScalar.save') 
    model.load_model('./conModel/2022-05-17-01-17-model.h5')
    
    # 2. 把資料轉成model需求ㄉinput格式
    conData = to_supervised(conData, n_in=23, n_out=1, var='consumption')
    conData = scalar.fit_transform(conData)
    X = conData[0, :] # 測試看看準不準==
    # X = conData[-1, :]
    X = X.reshape(1, 24)  

    # 3. 預測yo
    conPredict = model.predict(X)
    invCon = scalar.inverse_transform(conPredict)
    invCon = invCon.reshape(24,)
    # roundCon = [round(num, 2) for num in invCon]
    # print('Consumption')
    # print(roundCon)
    
    # 輸出預測結果
    data = []
    NextTime = time[-1] + timedelta(hours=1)
    for i in range(1, 25, 1):
        bias = invCon[i-1]-invGen[i-1]
        if bias >= 0:   # 需要買電
            data.append([NextTime.strftime('%Y-%m-%d %H:%M:%S'), "buy", 2.5, round(bias, 2)])
        else:   # 有多的電ㄛ
            data.append([NextTime.strftime('%Y-%m-%d %H:%M:%S'), "sell", 3, round(-bias, 2)])
        NextTime += timedelta(hours=1)
    output(args.output, data)   
    