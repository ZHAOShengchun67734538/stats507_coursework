import time
import numpy as np
from numpy import abs
from numpy import log
from numpy import sign
from factor_util import *


class Alpha101(object):
    def __init__(self, df_data):
        self.close = df_data['close']
        self.returns = df_data['returns']
        self.volume = df_data['volume']
        self.open = df_data['open']
        self.high = df_data['high']
        self.low = df_data['low']
        self.vwap = df_data['vwap']

    def alpha001(self):
        inner = self.close.copy()
        inner[self.returns < 0] = stddev(self.returns, 20)
        return ts_argmax(inner ** 2, 5).rank(axis=1, pct=True) - 0.5

    def alpha002(self):
        df = -1 * correlation(rank(delta(log(self.volume), 2)), rank((self.close - self.open) / self.open), 6)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha003(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha004(self):
        return -1 * ts_rank(rank(self.low), 9)


    def alpha006(self):
        df = -1 * correlation(self.open, self.volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha007(self):
        adv20 = sma(self.volume, 20)
        alpha = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha

    def alpha008(self):
        return -1 * (rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                           delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))


    def alpha011(self):
        return ((rank(ts_max((self.vwap - self.close), 3)) + rank(ts_min((self.vwap - self.close), 3))) * rank(
            delta(self.volume, 3)))

    def alpha012(self):
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    def alpha013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha014(self):
        df = correlation(self.open, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * rank(delta(self.returns, 3)) * df

    def alpha015(self):
        df = correlation(rank(self.high), rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_sum(rank(df), 3)

    def alpha016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def alpha017(self):
        adv20 = sma(self.volume, 20)
        return -1 * (rank(ts_rank(self.close, 10)) *
                     rank(delta(delta(self.close, 1), 1)) *
                     rank(ts_rank((self.volume / adv20), 5)))

    def alpha018(self):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank((stddev(abs((self.close - self.open)), 5) + (self.close - self.open)) +
                          df))

    def alpha019(self):
        return ((-1 * sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) *
                (1 + rank(1 + ts_sum(self.returns, 250))))

    def alpha020(self):
        return -1 * (rank(self.open - delay(self.high, 1)) *
                     rank(self.open - delay(self.close, 1)) *
                     rank(self.open - delay(self.low, 1)))


    def alpha022(self):
        df = correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * delta(df, 5) * rank(stddev(self.close, 20))



    def alpha025(self):
        adv20 = sma(self.volume, 20)
        return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close)))


    def alpha026(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_max(df, 3)

    def alpha028(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return scale(((df + ((self.high + self.low) / 2)) - self.close))

    def alpha029(self):
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))), 5) +
                ts_rank(delay((-1 * self.returns), 6), 5))

    def alpha030(self):
        delta_close = delta(self.close, 1)
        inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)

    def alpha032(self):
        return scale((sma(self.close, 7) - self.close)) + (20 * scale(correlation(self.vwap, delay(self.close, 5), 230)))


    def alpha033(self):
        return rank(-1 + (self.open / self.close))

    def alpha034(self):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return rank(2 - rank(inner) - rank(delta(self.close, 1)))

    def alpha035(self):
        return ((ts_rank(self.volume, 32) *
                 (1 - ts_rank(self.close + self.high - self.low, 16))) *
                (1 - ts_rank(self.returns, 32)))

    def alpha037(self):
        return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)

    def alpha038(self):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return -1 * rank(ts_rank(self.close, 10)) * rank(inner)

    def alpha040(self):
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    def alpha041(self):
        return pow((self.high * self.low), 0.5) - self.vwap

    def alpha042(self):
        return rank((self.vwap - self.close)) / rank((self.vwap + self.close))

    def alpha043(self):
        adv20 = sma(self.volume, 20)
        return ts_rank(self.volume / adv20, 20) * ts_rank((-1 * delta(self.close, 7)), 8)

    def alpha044(self):
        df = correlation(self.high, rank(self.volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

    def alpha052(self):
        return (((-1 * delta(ts_min(self.low, 5), 5)) *
                 rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))) * ts_rank(self.volume, 5))


    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)

    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return - ((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10))))

    def alpha101(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)

    def calculate(self):
        alpha001 = self.alpha001()
        alpha002 = self.alpha002()
        alpha003 = self.alpha003()
        alpha004 = self.alpha004()
        alpha006 = self.alpha006()
        alpha007 = self.alpha007()
        alpha008 = self.alpha008()
        alpha011 = self.alpha011()
        alpha012 = self.alpha012()
        alpha013 = self.alpha013()
        alpha014 = self.alpha014()
        alpha015 = self.alpha015()
        alpha016 = self.alpha016()
        alpha017 = self.alpha017()
        alpha018 = self.alpha018()
        alpha019 = self.alpha019()
        alpha020 = self.alpha020()
        alpha022 = self.alpha022()
        alpha025 = self.alpha025()

        alpha026 = self.alpha026()
        alpha028 = self.alpha028()
        alpha029 = self.alpha029()
        alpha030 = self.alpha030()
        alpha032 = self.alpha032()
        alpha033 = self.alpha033()
        alpha034 = self.alpha034()
        alpha035 = self.alpha035()
        alpha037 = self.alpha037()
        alpha038 = self.alpha038()
        alpha040 = self.alpha040()
        alpha041 = self.alpha041()
        alpha042 = self.alpha042()
        alpha043 = self.alpha043()
        alpha044 = self.alpha044()
        alpha052 = self.alpha052()
        alpha053 = self.alpha053()
        alpha054 = self.alpha054()
        alpha060 = self.alpha060()
        alpha101 = self.alpha101()

        alpha_dict = {'alpha001': alpha001, 'alpha002': alpha002, 'alpha003': alpha003, 'alpha004': alpha004,
                       'alpha006': alpha006, 'alpha007': alpha007, 'alpha008': alpha008,
                       'alpha011': alpha011, 'alpha012': alpha012, 'alpha013': alpha013, 'alpha014': alpha014,
                       'alpha015': alpha015, 'alpha016': alpha016, 'alpha017': alpha017, 'alpha018': alpha018,
                       'alpha019': alpha019, 'alpha020': alpha020, 'alpha022': alpha022, 'alpha025': alpha025,
                       'alpha026': alpha026, 'alpha028': alpha028, 'alpha029': alpha029, 'alpha030': alpha030,
                       'alpha032': alpha032, 'alpha033': alpha033, 'alpha034': alpha034, 'alpha035': alpha035,
                       'alpha037': alpha037, 'alpha038': alpha038, 'alpha040': alpha040, 'alpha041': alpha041,
                       'alpha042': alpha042, 'alpha043': alpha043, 'alpha044': alpha044, 'alpha052': alpha052,
                       'alpha053': alpha053, 'alpha054': alpha054, 'alpha060': alpha060,
                       'alpha101': alpha101,
                           }
        factors = pd.concat(alpha_dict, axis=1)
        factors.index.name = 'date'
        factors.columns.names = ['alpha', 'symbol']
        return factors







