import pandas as pd
import numpy as np

rank_df = pd.read_csv('2_rank_df.csv', index_col=0)
char_df = pd.read_csv('1_char_df.csv', index_col=0)
rank_df['bm'] = char_df['BM']
rank_df['size'] = char_df['Size']
# char_list = ['BM', 'Size']
char_list = ['BM']


def fun(subdf):
    total_mkt_cap = subdf['size'].sum()

    def fun2(subdf):
        mkt_cap_dist = subdf['size'].sum() / total_mkt_cap
        mkt_cap = subdf['size'].mean()
        bm = subdf['bm'].mean()
        stock_num = subdf.shape[0]
        series = subdf.iloc[0, :].copy()
        series['mkt_cap_dist'] = mkt_cap_dist
        series['mkt_cap'] = mkt_cap
        series['bm'] = bm
        series['stock_num'] = stock_num
        return series[['mkt_cap_dist', 'mkt_cap', 'bm','stock_num']]

    return subdf.groupby(char_list).apply(fun2)


tmp = rank_df.groupby(['date']).apply(fun)


def fun3(subdf):
    mkt_cap_dist = subdf.mkt_cap_dist.mean()
    mkt_cap = subdf.mkt_cap.mean()
    bm = subdf.bm.mean()
    stock_num = subdf.stock_num.mean()
    series = subdf.iloc[0, :].copy()
    series['mkt_cap_dist'] = mkt_cap_dist
    series['mkt_cap'] = mkt_cap
    series['bm'] = bm
    series['stock_num'] = stock_num
    return series[['mkt_cap_dist', 'mkt_cap', 'bm','stock_num']]


tmp = tmp.groupby(char_list).apply(fun3)
print(tmp.unstack(0))