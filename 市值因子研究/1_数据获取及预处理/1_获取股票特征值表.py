from six import BytesIO
import pandas as pd
import numpy as np
from jqfactor import get_factor_values


def initialize(context):
    run_monthly(get_Size_df, -1, time='17:00', force=False)
    g.mkt_cap = 'market_cap'
    g.char_df = pd.DataFrame(columns=['code', g.mkt_cap, 'date'])


def get_Size_df(context):
    stock_df = get_all_securities(types=['stock'], date=context.current_dt)
    current_data = get_current_data()
    avai_stock_list = [
        stock for stock in stock_df.index
        if not current_data[stock].paused and not current_data[stock].is_st
        and not '退' in current_data[stock].name
    ]
    factor_data = get_factor_values(securities=avai_stock_list,
                                    factors=[g.mkt_cap],
                                    end_date=context.current_dt,
                                    count=1)
    ME_df = factor_data[g.mkt_cap].T
    ME_df.rename(columns={ME_df.columns[0]: g.mkt_cap}, inplace=True)
    ME_df.dropna(inplace=True)
    # 前后交集去除新上市公司及后续变质公司(ST,退市等)
    ME_df['date'] = context.current_dt.date()
    ME_df = ME_df.reset_index(drop=False)
    g.char_df = g.char_df.append(ME_df)


def on_strategy_end(context):
    write_file('mkt_cap_monthly_reb_df.csv', g.char_df.to_csv(), append=False)