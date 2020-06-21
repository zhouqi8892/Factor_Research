import pandas as pd
import numpy as np
import statsmodels.api as sm
from jqfactor import get_factor_values
from dateutil.relativedelta import relativedelta
log.set_level('order', 'error')
'''为研究因子有效与横截面特征值回归，因子风险溢价显著性的关系'''


def initialize(context):  #初始化
    set_benchmark('000300.XSHG')
    set_order_cost(OrderCost(open_tax=0,
                             close_tax=0.001,
                             open_commission=0.0002,
                             close_commission=0.0002,
                             close_today_commission=0,
                             min_commission=5),
                   type='stock')
    set_slippaged(PriceRelatedSlippage(0.00246), type='stock')
    set_option('use_real_price', True)
    run_monthly(before_market_open, 1, time='9:00')
    g.char_info_dict = {'size': {'fun': lambda x: x, 'lag': 1}}

    g.char_list = list(g.char_info_dict.keys())
    g.max_lag = max(
        [item_dict['lag'] for item_dict in g.char_info_dict.values()])
    g.unit = '1M'
    g.window_research_list = [30, 36]
    g.window = max(g.window_research_list)

    g.reg_summary = pd.DataFrame()


def before_market_open(context):
    '''盘前生成待交易组合列表'''
    def market_df_gen(window, max_lag, unit):
        '''筛除停牌ST退市股票，对其余股票拿到历史成交价格'''
        count = max(window + max_lag, window + 1)
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
                return df

        market_df = market_df.groupby('code',
                                      group_keys=False).apply(return_cal)
        market_df.dropna(how="all",
                         inplace=True)  # 个别天apply出来把未通过判断的subdf输出全nan
        return market_df

    market_df = market_df_gen(g.window, g.max_lag, g.unit)

    #---------------------------factors -----------------------------
    def char_df_gen(market_df):
        date_list = market_df.date.unique()
        char_df = pd.DataFrame()
        for date in date_list:
            code_list = list(market_df.code[market_df.date == date].values)
            result_dict = get_factor_values(securities=code_list,
                                            factors=g.char_list,
                                            start_date=date,
                                            end_date=date)
            char_date_df = pd.DataFrame()
            char_date_df['code'] = code_list
            char_date_df['day'] = date
            for i in g.char_list:
                char_date_df[i] = result_dict[i].iloc[0].values
            char_df = char_df.append(char_date_df)
        char_df['day'] = pd.to_datetime(char_df['day'])
        return char_df

    char_df = char_df_gen(market_df)

    #---------------------------------merge_df--------------------------
    def market_df_integrate_gen(market_df, char_df):
        '''合并char_Df和market_df'''
        market_df_integrate = pd.merge(market_df,
                                       char_df,
                                       'left',
                                       left_on=['pd_datetime', 'code'],
                                       right_on=['day', 'code'])
        market_df_integrate = market_df_integrate.groupby(
            'code', group_keys=False).apply(lambda subdf: subdf.apply(
                lambda x: x.shift(g.char_info_dict[x.name]['lag']).iloc[max(
                    1, g.max_lag):]
                if x.name in g.char_list else x.iloc[max(1, g.max_lag):]))

        def drop_stock_with_nan(df):
            if not pd.isna(df).values.any():
                return df

        market_df_integrate = market_df_integrate.groupby(
            'code', group_keys=False).apply(drop_stock_with_nan)
        return market_df_integrate

    market_df_integrate = market_df_integrate_gen(market_df, char_df)

    # ---------------------------reg result------------------------------------
    def reg_result_df_gen():
        '''横截面回归得到特征loadings，取平均作本月特征估计值并计算T值，
        可考虑李斌ML方法每次重构时选择最优模型的回归结果'''
        market_df_integrate['const'] = 1

        # 是否加常数项回归？ 参考模型

        def ols_params(data, xcols, ycol):
            return sm.OLS(data[ycol], data[xcols]).fit().params[:]

        ycol = 'return'
        xcols = g.char_list + ['const']

        res = market_df_integrate.groupby(['year', 'month']).apply(ols_params,
                                                                   ycol=ycol,
                                                                   xcols=xcols)

        # 不能直接参考date来group，获取特征值的日期并非全部一致（停牌致月最后交易日提前）
        def fun_cal(x):
            tmp_se = pd.Series()
            for window in g.window_research_list:
                tmp = x[-window:]
                se = pd.Series(
                    [
                        np.mean(tmp),
                        np.sqrt(len(tmp)) * np.mean(tmp) / np.std(tmp)
                    ],
                    index=['estimator_%s' % window,
                           'T_value_%s' % window])
                tmp_se = tmp_se.append(se)
            return tmp_se
            # return tmp_df

        reg_result_df = res.iloc[:, :-1].apply(fun_cal).unstack()
        reg_result_df.name = context.current_dt.strftime('%Y-%m-%d')
        return reg_result_df

    reg_result_df = reg_result_df_gen()
    g.reg_summary = g.reg_summary.append(reg_result_df)
    log.info(g.reg_summary)


def on_strategy_end(context):
    '''在回测结束时调用'''
    g.reg_summary.columns = pd.MultiIndex.from_tuples(
        g.reg_summary.columns, names=['char', 'result_item'])
    write_file('reg_summary_30_36.csv', g.reg_summary.to_csv(), append=False)
