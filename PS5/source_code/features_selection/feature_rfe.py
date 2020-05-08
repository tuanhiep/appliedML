import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def my_RFE(X,Y):

    model = LogisticRegression()
    rfe = RFE(model, 10)
    fit = rfe.fit(X, Y)
    #print("Num Features: %d" % fit.n_features_)
    #print("Selected Features: %s" % fit.support_)
    print("* Print out 10 best features")
    print(X.columns[np.where(fit.support_==True)[0]].values.tolist())
    #print("Feature Ranking: %s" % fit.ranking_)
