from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def myFeature_RFE(X,Y):

    # feature extraction
    print('#----------------------------- Feature Extraction : RFE  -------------------------------------#')

    model = LogisticRegression()
    rfe = RFE(model, 10)
    fit = rfe.fit(X, Y)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
