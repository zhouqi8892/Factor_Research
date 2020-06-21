import pandas as pd
import numpy as np
import statsmodels.api as sm
import Order
from dateutil.relativedelta import relativedelta
import pandas as pd
from six import BytesIO

log.set_level('order', 'error')
'''为研究因子有效与个股时间序列回归，因子loading的关系'''


def initialize(context):
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
    run_monthly(before_market_open, 1,time='7:00',force=False)
    run_monthly(trade, 1,time='9:30',force=False)

    body=read_file('SMB_factor_portfolio_5x1.csv')
    g.SMB_df=pd.read_csv(BytesIO(body))
    g.ts_window = 6

def before_market_open(context):
    def market_df_gen():
        '''筛除停牌ST退市股票，对其余股票拿到历史成交价格'''
        count = g.ts_window+1
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
                             '1M', ['date', 'close'],
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
                return df.iloc[1:]

        market_df = market_df.groupby('code',
                                      group_keys=False).apply(return_cal)
        market_df.dropna(how="all",inplace=True) # 个别天apply出来把未通过判断的subdf输出全nan
        return market_df

    market_df = market_df_gen()
    def ts_reg(market_df):
        tmp_df = market_df.copy()
        tmp_df['const'] = 1
        tmp_df = pd.merge(tmp_df,g.SMB_df,'left',['year','month'])
        def ols_params(data, xcols, ycol):
            result = sm.OLS(data[ycol], data[xcols]).fit()
            return pd.Series([result.params.iloc[0],result.tvalues.iloc[0]],index=['beta','t_value'])

        ycol = 'return_x'
        xcols =['return_y','const']
        res = tmp_df.groupby(['code']).apply(ols_params,
                                              ycol=ycol,
                                              xcols=xcols)
        return res
    ts_reg_df = ts_reg(market_df)
    # 是否考虑t值，组件一个显著的一个全部的
    sig_stock_df = ts_reg_df[np.abs(ts_reg_df.t_value)>2]
    g.long_list=list(sig_stock_df[sig_stock_df.beta >2].index)
    log.info('组合个股数量%s'%len(g.long_list))
def trade(context):
    Order.position_adjust(g.long_list, 1, 'value_weighted', context)
