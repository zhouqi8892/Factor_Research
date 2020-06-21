from six import BytesIO
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from calendar import monthrange

def initialize(context):
    run_monthly(bars, -1, time='17:00', force=False)
    run_monthly(get_rank_df, -1, time='17:00', force=False)
    run_monthly(stock_return_df_gen, -1, time='17:00', force=False)
    run_monthly(save_result, -1, time='17:00', force=False)
    g.sort_group_dict = {'market_cap': 5}
    g.char_list = list(g.sort_group_dict.keys())

def bars(context):
    body = read_file('mkt_cap_monthly_reb_df.csv')
    char_df = pd.read_csv(BytesIO(body), index_col=0)  
    price_df = get_bars(list(char_df.code.unique()),
                        count=999,
                        include_now=True,
                        unit='1M',
                        end_dt=char_df['date'].iloc[-1],
                        fields=['date', 'close'],
                        df=True,
                        fq_ref_date=datetime.now())
    price_df['year'] = price_df['date'].apply(lambda x: x.year)
    price_df['month'] = price_df['date'].apply(lambda x: x.month)
    def return_cal(subdf):
        subdf.sort_values('date',inplace=True)
        subdf['return'] = subdf['close'].diff() / subdf['close'].shift()
        return subdf
    price_df = price_df.groupby(level=0).apply(return_cal)
    price_df.index = price_df.index.droplevel(1)
    price_df.index.name='code'
    price_df.reset_index(inplace=True,drop=False)
    price_df.set_index(['code','year','month'],inplace=True)
    g.price_df = price_df

def get_rank_df(context):
    '''待考察break_point对象的影响(NYSE for US stocks),目前是全市场'''
    body = read_file('mkt_cap_monthly_reb_df.csv')
    g.char_df = pd.read_csv(BytesIO(body), index_col=0)
    rank_df = g.char_df.copy()

    def sort(subdf):
        break_point_dict = {}
        for char in g.char_list:
            break_point_dict[char] = subdf[char].quantile([
                1 / g.sort_group_dict[char] * i
                for i in range(1, g.sort_group_dict[char])
            ])
        subdf = subdf[g.char_list]

        def assign_group_num(series, break_point_dict):
            def fun(x, break_point_list, group_num):
                # lebel1 is the highest
                return group_num - len(break_point_list[break_point_list < x])

            item = series.name
            group_num = g.sort_group_dict[item]
            series = series.apply(
                lambda x: fun(x, break_point_dict[item].values, group_num))
            return series

        return subdf.apply(assign_group_num,
                           break_point_dict=break_point_dict,
                           axis=0)

    tmp_df = rank_df.groupby('date', group_keys=False).apply(sort)
    rank_df.update(tmp_df)
    g.rank_df = rank_df


def stock_return_df_gen(context):
    rank_df = g.rank_df
    char_df = g.char_df.set_index('code')

    def get_return(subdf):
        start_date = datetime.strptime(subdf['date'].iloc[0], '%Y-%m-%d')
        if subdf['date'].iloc[0] == rank_df['date'].iloc[-1]:
            return
        else:
            tmp_date = (start_date +relativedelta(months=1))
            end_date = datetime(tmp_date.year,tmp_date.month,monthrange(tmp_date.year, tmp_date.month)[1])
            subdf['mkt_cap'] = char_df[g.char_list[0]][char_df['date'] ==
                                              subdf['date'].iloc[0]].values
            def return_cal(df):
                #剔除过程中整月停牌的股票'
                idx = (df['code'].iloc[0],end_date.year,end_date.month)
                if idx in  g.price_df.index:
                    next_month_df = g.price_df.loc[[idx],:]
                    next_month_df[g.char_list[0]]=df[g.char_list[0]].iloc[0]
                    next_month_df['mkt_cap'] = df['mkt_cap'].iloc[0]
                    return next_month_df

            return_df = subdf.groupby('code',group_keys=False).apply(return_cal)
            return_df.reset_index(inplace=True,drop=False)
            return return_df
    stock_return_df = rank_df.groupby('date',
                                          group_keys=False).apply(get_return)
    del stock_return_df['close']
    print(stock_return_df.columns)
    print(stock_return_df)
    stock_return_df.reset_index(inplace=True)                          
    g.stock_return_df = stock_return_df




def save_result(context):
    write_file('market_cap_rank_df.csv', g.rank_df.to_csv(), append=False)
    write_file('market_cap_stock_return_plus_rank_df.csv',
               g.stock_return_df.to_csv(),
               append=False)