import pandas as pd
import numpy as np
import statsmodels.api as sm
import Order
from jqfactor import get_factor_values
from dateutil.relativedelta import relativedelta
log.set_level('order', 'error')
from six import BytesIO


def initialize(context):  #初始化
    set_benchmark('000300.XSHG')
    set_order_cost(OrderCost(open_tax=0,
                             close_tax=0.001,
                             open_commission=0.0002,
                             close_commission=0.0002,
                             close_today_commission=0,
                             min_commission=5),
                   type='stock')
    set_slippage(PriceRelatedSlippage(0.00246), type='stock')
    set_option('use_real_price', True)
    run_monthly(before_market_open, 1, time='9:00')
    g.char_info_dict = {
        'size': {
            'fun': lambda x: x,
            'lag': 1
        },
    }

    g.char_list = list(g.char_info_dict.keys())
    g.max_lag = max(
        [item_dict['lag'] for item_dict in g.char_info_dict.values()])
    g.unit = '1M'
    g.cs_window = 1
    g.ts_window = 3 # min=2
    g.summary_df = pd.DataFrame()
    body=read_file('SMB_factor_portfolio_5x1.csv')
    g.SMB_df=pd.read_csv(BytesIO(body))

def before_market_open(context):
    
    def market_df_gen(cs_window,ts_window, max_lag, unit):
        '''筛除停牌ST退市股票，对其余股票拿到历史成交价格'''
        count = cs_window + max_lag + ts_window
        # 利用df的时间获取char，考虑lag需要shift的空间，lag=0时，计算return也需要1次shift
        stock_df = get_all_securities(['stock'], context.current_dt)
        current_data = get_current_data()
        avai_stock_list = [
            stock for stock in stock_df.index
            if not current_data[stock].paused and not current_data[stock].is_st
            and not '退' in current_data[stock].name
        ]
        # 停牌、ST、退市 股票筛除
        market_df = get_bars(avai_stock_list,
                             count,
                             unit, ['date', 'close'],
                             fq_ref_date=context.current_dt,
                             df=True)
        market_df['year'] = market_df['date'].apply(lambda x: x.year)
        market_df['month'] = market_df['date'].apply(lambda x: x.month)
        market_df['pd_datetime'] = pd.to_datetime(market_df.date)
        market_df.index.names = ['code', 'idx']
        market_df.index = market_df.index.droplevel(1)
        market_df.reset_index(inplace=True)

        def return_cal(df):
            '''计算return并剔除第一个nan'''
            judge_date = df.date.iloc[0] + relativedelta(months=count)
            if df.shape[
                    0] == count and judge_date.year == context.current_dt.year and judge_date.month == context.current_dt.month:
                df['return'] = df['close'].diff() / df['close'].shift()
                # 月价格不连续（中间有整月停牌）的股票剔除
                # 第一行return nan剔除
                return df.iloc[1:]

        market_df = market_df.groupby('code',
                                      group_keys=False).apply(return_cal)
        market_df.dropna(how="all",inplace=True) # 个别天apply出来把未通过判断的subdf输出全nan
        return market_df

    market_df = market_df_gen(g.cs_window, g.ts_window,g.max_lag, g.unit)    

    #----------------------------------TS regression---------------------------
    def ts_reg_loading_gen(market_df,factor_df,ts_window):
        def ts(subdf):
            def RollOls_params(indices, ycol, xcols):
                roll_df=subdf.loc[indices]
                result = sm.OLS(roll_df[ycol], roll_df[xcols]).fit()
                params = result.params
                tmp_para_result_df.loc[indices[-1],xcols] = params
                return 0
            tmp_para_result_df = subdf.copy()
            subdf['index'].rolling(ts_window).apply(lambda indices:RollOls_params(indices, ycol=ycol,xcols=xcols),raw=True)
            # 输出
            return tmp_para_result_df.iloc[ts_window-1:]
            
        merge_df = pd.merge(market_df,factor_df,'left',['year','month'])
        merge_df['const'] = 1
        merge_df['index'] = merge_df.index
        ycol = 'return_x'
        xcols = ['return_y','const']
        return merge_df.groupby('code',group_keys=False).apply(ts)

    merge_loading_df = ts_reg_loading_gen(market_df,g.SMB_df,g.ts_window)
    #-----------------------------lag process--------------------------------
    def lag_gen(merge_loading_df):    
        loading_lag_df = merge_loading_df.groupby(
            'code', group_keys=False).apply(lambda subdf: subdf.apply(
                lambda x: x.shift(g.char_info_dict[x.name]['lag']).iloc[g.max_lag:]
                if x.name in g.char_list else x.iloc[g.max_lag:]))

        return loading_lag_df

    market_df_integrate = lag_gen(merge_loading_df)
    # ---------------------------cs reg------------------------------------
    def reg_result_df_gen(market_df_integrate):
        '''横截面回归得到特征loadings，取平均作本月特征估计值并计算T值，
        可考虑李斌ML方法每次重构时选择最优模型的回归结果'''
        market_df_integrate['const'] = 1

        # 是否加常数项回归？ 参考模型

        def ols_params(data, xcols, ycol):
            return sm.OLS(data[ycol], data[xcols]).fit().params[:]

        ycol = 'return_x'
        xcols = ['return_y','const']

        res = market_df_integrate.groupby(['year', 'month']).apply(ols_params,
                                                                   ycol=ycol,
                                                                   xcols=xcols)
        # 不能直接参考date来group，获取特征值的日期并非全部一致（停牌致月最后交易日提前）
        reg_result_df = res.apply(lambda x: pd.Series(
            [np.mean(x), np.sqrt(len(x)) * np.mean(x) / np.std(x)],
            index=['estimator', 'T_value'])).iloc[:, :-1]
        return reg_result_df
        
    reg_result_df = reg_result_df_gen(market_df_integrate)
    tmp_df = reg_result_df.T
    tmp_df.index=[context.current_dt.strftime('%Y-%m-%d')]
    print(tmp_df)
    g.summary_df = g.summary_df.append(tmp_df)
    return


def on_strategy_end(context):
    '''在回测结束时调用'''
    write_file(f'summary_cs{g.cs_window}_ts{g.ts_window}_lag{g.max_lag}_df.csv', g.summary_df.to_csv(), append=False)
