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
from imp import reload
import futurelogistic.futurelogistic
reload(futurelogistic.futurelogistic)
import InvestBase
reload(InvestBase)
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

minp = 0.005
pnumber = 4
label = 'xgbtest'
futuremodel = futurelogistic.futurelogistic.FutureLogistic(minp, pnumber, label)
self = futuremodel

###############################logistic strategy:classification#####################
testlen = 60
ntrain = 12
lengths = [1,3,5,9,15,30]
timesteps = 20
day = 2
attr = 'cr'
attry = 'roo'
modellabel = 'lr'
readfile = False
hsma = futuremodel.lr_cls(testlen, ntrain, lengths, timesteps, day, attr, attry, 
                           modellabel, readfile)
pr = 0.5
fee = 0.0004
hsmaratio = futuremodel.hsmadata_daycode_lsr(hsma, day, pr, fee)
tradestat = InvestBase.tradestat_portfolio(hsmaratio)
plt.plot(hsmaratio.ratio)

####################################################################
###############################FutureMinute test##############################
####################################################################,
import sys
sys.path.append("Test")
from imp import reload
import futurelogistic.futureminutelogistic
reload(futurelogistic.futureminutelogistic)
import InvestBase
reload(InvestBase)
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

minp = 0.005
pnumber = 4
label = 'logistic'
futuremodel = futurelogistic.futurelogistic.FutureLogistic(minp, pnumber, label)
self = futuremodel

###############################logistic strategy:classification#####################
testlen = 60
ntrain = 12
lengths = [1,3,5,9,15,30]
timesteps = 20
day = 2
attr = 'cr'
attry = 'roo'
modellabel = 'lr'
readfile = False
hsma = futuremodel.lr_cls(testlen, ntrain, lengths, timesteps, day, attr, attry, 
                           modellabel, readfile)
pr = 0.5
fee = 0.0004
hsmaratio = futuremodel.hsmadata_daycode_lsr(hsma, day, pr, fee)
tradestat = InvestBase.tradestat_portfolio(hsmaratio)
plt.plot(hsmaratio.ratio)

#######################################################################
###############################Trading Strategies######################
#######################################################################
