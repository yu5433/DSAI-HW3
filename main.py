import imp
from multiprocessing.dummy import JoinableQueue
import pandas as pd
import numpy as np
import argparse
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


if __name__ == "__main__":
    args = config()
    # tmpConFile = 'consumption.csv'
    # tmpGenFile = 'generation.csv'

    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    Condata = pd.read_csv(args.consumption, parse_dates=['time'], date_parser=custom_date_parser, index_col=0, squeeze=True)
    Gendata = pd.read_csv(args.generation, parse_dates=['time'], date_parser=custom_date_parser, index_col=0, squeeze=True)
    data = []
    time = Condata.index
    print(time[-1] + timedelta(hours=1))
    NextTime = time[-1] + timedelta(hours=1)
    if Gendata[-1] < Condata[-1]:
        # means need more electricuty
        data.append([NextTime.strftime('%Y-%m-%d %H:%M:%S'), "buy", 2.5, 3])
        data.append([NextTime.strftime('%Y-%m-%d %H:%M:%S'), "sell", 3, 3])
    else:
        data.append([NextTime.strftime('%Y-%m-%d %H:%M:%S'), "sell", 3, 5])
        data.append([NextTime.strftime('%Y-%m-%d %H:%M:%S'), "buy", 2.36, 3])
    test = Condata.index

    output(args.output, data)
