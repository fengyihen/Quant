# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""

import sys
sys.path.append("Test")
from imp import reload
import FutureDay
reload(FutureDay)
import CTA.CTADay
reload(CTA.CTADay)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import tree

#from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1


class CTADaySklearn(FutureDay.Future):
    def __init__(self, timestep, pnumber, label):
        super(CTADaySklearn, self).__init__(pnumber, label)
        self.timestep = timestep
        self.hsmadata_raw_x = self.hsmadata_raw_x(timestep)

    #构造单个商品的x
    def indexcloser1day_Long(self, code, mrlist, crlist):

        hsma_raw_x = self.hsmadata_raw_x[self.hsmadata_raw_x.code ==
                                         code].copy()
        hsma = self.hsmaall[self.hsmaall.code == code].copy()

        mrcry = pd.DataFrame()
        for d in hsma_raw_x.date:
            print(d)
            for mr in mrlist:
                print(mr)
                for cr in crlist:
                    hsma = hsma[hsma.date > d]
                    if hsma.shape[0] < 30:
                        continue
                    hsma = hsma.iloc[0:30, ].copy()
                    fee = self.feelist[code]
                    hsmatrade = CTA.CTADay.indexcloser1day_Long(hsma, mr, \
                                                                cr, fee)
                    if hsmatrade.shape[0] == 0:
                        continue
                    temp = pd.DataFrame({
                        'date': d,
                        'mr': mr,
                        'cr': cr,
                        'ratio': hsmatrade.traderatio.sum()
                    },
                                        index=[0])
                    mrcry = pd.concat([mrcry, temp])

        hsmadata = pd.merge(hsma_raw_x, mrcry)
        hsmadata.drop(['code', 'date'], axis=1, inplace=True)

        return hsmadata

    def randomforestregressor(self, code, mrlist, crlist, testlen, ntrain,
                              ntrees, nodes):

        hsmadata = self.indexcloser1day_Long(code, mrlist, crlist)
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] < dates[i * testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen]) & (
                hsmadata['date'] < dates[(i + 1) * testlen])].copy()

            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            traindatay = traindata['ratio']
            testdatax = testdata[traindatax.columns]

            treemodel = RandomForestRegressor(
                n_estimators=ntrees,
                min_samples_split=nodes * 2,
                min_samples_leaf=nodes)
            treemodel.fit(traindatax, traindatay)
            testdata['predratio'] = treemodel.predict(testdatax)

            hsma = pd.concat([hsma, testdata], ignore_index=True)
        hsma.to_hdf(
            'Test\\stocksklearn\\hsma_extratreesregressor_' + self.label +
            '.h5', 'hsma')

        return (hsma)
