import pandas as pd
import numpy as np
from joblib import Parallel, delayed, cpu_count
import statsmodels.api as sm


def window_test(date_window, char, df):
    date_list = list(df.sort_index().index.unique())
    window_return_df = pd.DataFrame(columns=np.sort(df[char].unique()))
    date_roll_list = [
        date_list[i:date_window + i]
        for i in range(len(date_list) - date_window + 1)
    ]
    for date_list in date_roll_list:
        stock_return_df = df.loc[date_list, :]

        def portfolio_return_cal(subdf):
            '''subdf --> 指定char分组下的指定年月的组合个股表格'''
            avg_return = (subdf['return'] *
                          (subdf['mkt_cap'] / subdf['mkt_cap'].sum())).sum()
            # value weighted
            portfolio_series = subdf.iloc[0, :].copy()
            # 借用模板，因为合成一个组合return，故只需要一行
            portfolio_series['return'] = avg_return

            return portfolio_series['return']

        portfolio_return_df = stock_return_df.groupby(
            [char, 'year', 'month']).apply(portfolio_return_cal)
        summary_return_df = portfolio_return_df.groupby(
            char, group_keys=False).apply(np.mean)
        summary_return_df.name = f'{date_list[-1][0]}/{date_list[-1][1]}'
        window_return_df = window_return_df.append(summary_return_df)

    # index代表对应年月的月末，为次月rebalance做参考

    def monotone_test(series):
        mono_increase_big_to_small = series.diff().iloc[1:] > 0
        mono_increace_small_to_big = series.diff(-1).iloc[:-1] > 0
        if all(mono_increase_big_to_small):
            return 'mono_de'
        elif all(mono_increace_small_to_big):
            return 'mono_in'
        else:
            result = sm.OLS(series.values,
                            list(zip(list(series.index),
                                     [1] * series.size))).fit()
            t_val = result.tvalues
            if t_val[0] > 2:
                return 'reg_de'
            elif t_val[0] < -2:
                return 'reg_in'
            else:
                return False
            # 考虑用回归斜率 收集不严格单调的时点

    monotone_series = window_return_df.apply(monotone_test, axis=1)
    monotone_series.name = f'window_{date_window}'
    print(date_window)
    return pd.DataFrame(monotone_series.values,
                        index=monotone_series.index,
                        columns=[monotone_series.name])


if __name__ == "__main__":

    df = pd.read_csv(
        r'Size独立\monthly_reb\流通市值\circulating_market_cap_stock_return_plus_rank_df.csv',
        index_col=0)
    df.set_index(['year', 'month'], inplace=True)
    char = 'circulating_market_cap'

    def concat_parallel(fun):
        results = Parallel(n_jobs=cpu_count(), prefer="processes")(
            delayed(fun)(date_window=date_window, char=char, df=df)
            for date_window in range(1, 184))
        return pd.concat(results, axis=1, sort=False)

    # concat_parallel(window_test)
    concat_parallel(window_test).to_csv('mono_result_monthly_sort.csv')
