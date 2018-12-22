# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""

import os
import sys
sys.path.append("Test")
from imp import reload
import FutureDay
reload(FutureDay)
import InvestBase
reload(InvestBase)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest


class FutureLogistic(FutureDay.Future):
    def lr_cls(self, testlen, ntrain, lengths, timesteps, day, attr, attry,
               modellabel, readfile):

        if attr == 'raw':
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        elif attr == 'ta':
            hsmadata_x = self.hsmadata_ta_x(lengths)
        elif attr == 'cr':
            hsmadata_x = self.hsmadata_cr_x(timesteps)
        else:
            print('Wrong Attr!')

        if attry == 'roc':
            hsmadata_y = self.hsmadata_roc(day)
        elif attry == 'roo':
            hsmadata_y = self.hsmadata_roo(day)
        else:
            print('Wrong Attr_y!')
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        filename = 'testresult\\futurexgboost\\hsma_xgb_cls_testlen' + \
            str(testlen) + '_attr' + str(attr) + '_attry' + str(attry) + \
            '_timesteps' + str(timesteps) + '_day' + str(day) + \
            '_' + modellabel + '_' + self.label + '.h5'

        if readfile:
            if os.path.exists(filename):
                hsma = pd.read_hdf(filename, 'hsma')
            else:
                hsma = pd.DataFrame()
        else:
            hsma = pd.DataFrame()

        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day - 1])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] < dates[
                                    (i + 1) * testlen])].copy()
            startdate = dates[i * testlen]
            enddate = testdata.date.max()
            if hsma.shape[0] > 0:
                if startdate <= hsma.date.max():
                    continue
            print(enddate)

            ###变换数据集成LSTM所需格式
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            testdatax = testdata[traindatax.columns]
            traindatay = traindata['ratio'].copy()
            traindatay[traindata['ratio'] >= 0] = 1
            traindatay[traindata['ratio'] < 0] = 0

            #加入变量筛选

            ###建模并预测
            ###xgboost sklearn api
            if modellabel == 'lr':
                xclas = LogisticRegression()  #objective='multi:softmax'
                xclas.fit(traindatax, traindatay)
                testdata['pred'] = xclas.predict(testdatax)
                testdata['prob_long'] = xclas.predict_proba(testdatax)[:, 1]
                testdata['prob_short'] = 1 - testdata['prob_long']
            else:
                pass

            if i == ntrain:
                hsma = testdata[[
                    'code', 'date', 'ratio', 'pred', 'prob_long', 'prob_short'
                ]].copy()
            else:
                hsma = pd.concat([
                    hsma, testdata[[
                        'code', 'date', 'ratio', 'pred', 'prob_long',
                        'prob_short'
                    ]]
                ],
                                 ignore_index=True)

            if readfile:
                hsma.to_hdf(filename, 'hsma')

        return (hsma)

    def lr_code_cls(self, testlen, ntrain, lengths, timesteps, day, attr,
                    attry, modellabel, readfile):

        if attr == 'raw':
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        elif attr == 'ta':
            hsmadata_x = self.hsmadata_ta_x(lengths)
        elif attr == 'cr':
            hsmadata_x = self.hsmadata_cr_x(timesteps)
        else:
            print('Wrong Attr!')

        if attry == 'roc':
            hsmadata_y = self.hsmadata_roc(day)
        elif attry == 'roo':
            hsmadata_y = self.hsmadata_roo(day)
        else:
            print('Wrong Attr_y!')
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        filename = 'testresult\\futurexgboost\\hsma_xgb_cls_testlen' + \
            str(testlen) + '_attr' + str(attr) + '_attry' + str(attry) + \
            '_timesteps' + str(timesteps) + '_day' + str(day) + \
            '_' + modellabel + '_' + self.label + '.h5'

        if readfile:
            if os.path.exists(filename):
                hsma = pd.read_hdf(filename, 'hsma')
            else:
                hsma = pd.DataFrame()
        else:
            hsma = pd.DataFrame()

        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day - 1])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] < dates[
                                    (i + 1) * testlen])].copy()
            startdate = dates[i * testlen]
            enddate = testdata.date.max()
            if hsma.shape[0] > 0:
                if startdate <= hsma.date.max():
                    continue
            print(enddate)

            ###建模并预测
            testpred = pd.DataFrame()
            for code in traindata.code.unique():
                traindata_c = traindata[traindata.code == code].copy()
                testdata_c = testdata[testdata.code == code].copy()
                if (testdata_c.shape[0] == 0) | (traindata_c.shape[0] <=
                                                 testdata_c.shape[0] * 3):
                    continue

                traindatax = traindata_c.drop(['date', 'code', 'ratio'], 1)
                traindatay = traindata_c['ratio'].copy()
                traindatay[traindata_c['ratio'] >= 0] = 1
                traindatay[traindata_c['ratio'] < 0] = 0

                if modellabel == 'lr':
                    xclas = LogisticRegression()  #objective='multi:softmax'
                    xclas.fit(traindatax, traindatay)
                    testdatax = testdata_c[traindatax.columns]
                    testdata_c['pred'] = xclas.predict(testdatax)
                    testdata_c['prob_long'] = xclas.predict_proba(
                        testdatax)[:, 1]
                    testdata_c['prob_short'] = 1 - testdata_c['prob_long']
                else:
                    pass
                testpred = pd.concat([testpred, testdata_c], ignore_index=True)

            if i == ntrain:
                hsma = testpred[[
                    'code', 'date', 'ratio', 'pred', 'prob_long', 'prob_short'
                ]].copy()
            else:
                hsma = pd.concat([
                    hsma, testpred[[
                        'code', 'date', 'ratio', 'pred', 'prob_long',
                        'prob_short'
                    ]]
                ],
                                 ignore_index=True)

            if readfile:
                hsma.to_hdf(filename, 'hsma')

        return (hsma)
