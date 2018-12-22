# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 00:26:24 2018

@author: fengy
"""
import sys
sys.path.append("Test")
from imp import reload
import pandas as pd
import numpy as np
import FutureDay
reload(FutureDay)


class CTADaySklearn(FutureDay.Future):
    def volpricebreak(self, code, length, m, fee):

        hsma = self.hsmaall[self.hsmaall.code == code].copy()
        hsma.index = range(0, hsma.shape[0])
        hsma['volp'] = (hsma.close - hsma.open) / (
            hsma.high - hsma.low + 1) * hsma.vol
        hsma['volp_sum'] = hsma.volp.rolling(length).sum()
        hsma['volp_mean'] = hsma.volp_sum.rolling(length).mean()
        hsma['volp_std'] = hsma.volp_sum.rolling(length).std()
        hsma['volp_up'] = hsma.volp_mean + hsma.volp_std * m
        hsma['volp_dn'] = hsma.volp_mean - hsma.volp_std * m

        hsma['LS'] = 'N'
        temp = hsma.volp_sum.shift(1) > hsma.volp_up.shift(1)
        hsma.loc[temp, 'LS'] = 'L'
        temp = hsma.volp_sum.shift(1) < hsma.volp_dn.shift(1)
        hsma.loc[temp, 'LS'] = 'S'
        hsma.loc[hsma.shape[0] - 1, 'LS'] = 'N'

        hsmatrade = pd.DataFrame()
        LSarray = np.where((hsma.LS == 'L') | (hsma.LS == 'S'))[0]
        if len(LSarray) == 0:
            return hsmatrade

        i = LSarray[0]
        while True:

            ls = hsma['LS'].iloc[i]
            opendate = hsma['date'].iloc[i]
            openprice = hsma['open'].iloc[i]

            if ls == 'L':
                jarray = np.where(
                    hsma.loc[i:(hsma.shape[0] - 1), 'volp_sum'] < hsma.loc[i:(
                        hsma.shape[0] - 1), 'volp_mean'])[0]
                if len(jarray) == 0:
                    j = hsma.shape[0] - 1
                else:
                    j = i + jarray[0] + 1
            if ls == 'S':
                jarray = np.where(
                    hsma.loc[i:(hsma.shape[0] - 1), 'volp_sum'] > hsma.loc[i:(
                        hsma.shape[0] - 1), 'volp_mean'])[0]
                if len(jarray) == 0:
                    j = hsma.shape[0] - 1
                else:
                    j = i + jarray[0] + 1

            closedate = hsma['date'].iloc[j]
            closeprice = hsma['open'].iloc[j]

            if ls == 'L':
                traderatio = closeprice / openprice - 1 - self.feelist[code]
            else:
                traderatio = 1 - closeprice / openprice - self.feelist[code]

            temp = pd.DataFrame({
                'LS': ls,
                'opendate': opendate,
                'openprice': openprice,
                'closedate': closedate,
                'closeprice': closeprice,
                'traderatio': traderatio,
                'tradetime': j - i,
            },
                                index=[0])

            hsmatrade = pd.concat([hsmatrade, temp])

            if any(LSarray > j):
                i = LSarray[LSarray > j][0]
            else:
                break

        hsmatrade.index = range(0, hsmatrade.shape[0])
        hsmatrade['ratio'] = hsmatrade.traderatio.cumsum()

        return hsmatrade

    def indexcloser1day_Long(self, code, mr, cr, fee):

        hsma = self.hsmaall[self.hsmaall.code == code].copy()
        hsma.index = range(0, hsma.shape[0])
        hsma['lastclose'] = hsma.close.shift(1)
        hsma['LS'] = 'N'
        temp = (hsma.open <= hsma.lastclose * (1 + mr)) & \
                (hsma.high >= hsma.lastclose * (1 + cr))
        hsma.loc[temp, 'LS'] = 'L'

        hsmatrade = pd.DataFrame()
        for i in hsma.index:

            if i == hsma.index.max():
                continue

            #开仓
            if hsma.loc[i, 'LS'] == 'L':
                opendate = hsma.loc[i, 'date']
                openprice = hsma.loc[i, 'lastclose'] * (1 + cr)
                closedate = hsma.loc[i + 1, 'date']
                closeprice = hsma.loc[i + 1, 'open']
                traderatio = closeprice / openprice - 1 - fee

                temp = pd.DataFrame({
                    'LS': 'L',
                    'opendate': opendate,
                    'closedate': closedate,
                    'openprice': openprice,
                    'closeprice': closeprice,
                    'traderatio': traderatio
                },
                                    index=[0])

                hsmatrade = pd.concat([hsmatrade, temp])

        return hsmatrade

    def indexcloser1day_Short(self, code, mr, cr, fee):

        hsma = self.hsmaall[self.hsmaall.code == code].copy()
        hsma.index = range(0, hsma.shape[0])
        hsma['lastclose'] = hsma.close.shift(1)
        hsma['LS'] = 'N'
        temp = (hsma.open >= hsma.lastclose * (1 - mr)) & \
                (hsma.low <= hsma.lastclose * (1 - cr))
        hsma.loc[temp, 'LS'] = 'S'

        hsmatrade = pd.DataFrame()
        for i in hsma.index:

            if i == hsma.index.max():
                continue

            #开仓
            if hsma.loc[i, 'LS'] == 'S':
                opendate = hsma.loc[i, 'date']
                openprice = hsma.loc[i, 'lastclose'] * (1 - cr)
                closedate = hsma.loc[i + 1, 'date']
                closeprice = hsma.loc[i + 1, 'open']
                traderatio = 1 - closeprice / openprice - fee

                temp = pd.DataFrame({
                    'LS': 'S',
                    'opendate': opendate,
                    'closedate': closedate,
                    'openprice': openprice,
                    'closeprice': closeprice,
                    'traderatio': traderatio
                },
                                    index=[0])

                hsmatrade = pd.concat([hsmatrade, temp])

        return hsmatrade
