# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""
####################################################################
###############################ALL test##############################
####################################################################,
import sys
sys.path.append("Test")
import numpy as np
import InvestBase
from imp import reload
import CTADaySklearn.CTADaySklearn
reload(CTADaySklearn.CTADaySklearn)
import pandas as pd
import matplotlib.pyplot as plt

####################################################################
###############################day test##############################
####################################################################,
timestep = 60
pnumber = 10
label = 'CTADaytest'
futuremodel = CTADaySklearn.CTADaySklearn.CTADaySklearn(
    timestep, pnumber, label)
self = futuremodel

####################################################################
###############################day test##############################
####################################################################,
code = 'rb000_day'
mrlist = np.arange(0.005, 0.055, 0.005)
crlist = np.arange(0.005, 0.055, 0.005)

mr = 0.0075
cr = 0.0095
fee = 0.0002
label = 'CTAtest'
futuremodel = CTAsklearn.CTAsklearn.CTAsklearn(cycle, label)
self = futuremodel

###############################logistic classifier#####################
#多品种共同建模
testlen = 20  # 等于day
ntrain = 36
llen = 200
slen = 20
tr = 0.002
hsma = futuremodel.logistic_dayin(testlen, ntrain, llen, slen, tr)
wtime = 10
tpr = 0.01
hsmatrade = futuremodel.hsmatrade(hsma, wtime, tpr)
portfolio = futuremodel.portfolio(hsmatrade)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

#测试结果统计
filename = 'Test\\futurekeras\\testresult\\hsma_lstm_cls_kerastest_LSTM1cls.h5'
hsma = pd.read_hdf(filename, 'hsma')
hsmaratio, portfolio = futuremodel.hsmadata_predp_r(hsma, lr)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

###############################tree classifier#####################
#多品种共同建模
testlen = 20  # 等于day
ntrain = 36
llen = 200
slen = 20
tr = 0.002
hsma = futuremodel.tree_dayin(testlen, ntrain, llen, slen, tr)
wtime = 10
tpr = 0.01
hsmatrade = futuremodel.hsmatrade(hsma, wtime, tpr)
portfolio = futuremodel.portfolio(hsmatrade)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

#测试结果统计
filename = 'Test\\futurekeras\\testresult\\hsma_lstm_cls_kerastest_LSTM1cls.h5'
hsma = pd.read_hdf(filename, 'hsma')
hsmaratio, portfolio = futuremodel.hsmadata_predp_r(hsma, lr)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)
