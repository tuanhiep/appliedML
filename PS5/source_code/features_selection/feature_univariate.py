from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy

def my_feature_univariate(X,Y):

    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(X, Y)
    # summarize scores
    numpy.set_printoptions(precision=3)
    #print("* Print out fit scores of features")
    #print(fit.scores_)
    print("* Print out 10 best features")
    fit.scores_.argsort()[-10:][::-1]
    print(X.columns[fit.scores_.argsort()[-10:][::-1]].values.tolist())
    #features = fit.transform(X)
