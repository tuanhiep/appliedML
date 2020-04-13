from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy

def myFeature_univariate(data2,X,Y):

    # feature extraction
    print('#----------------------------- Feature Extraction : Univariate  -------------------------------------#')
    
    test = SelectKBest(score_func=chi2, k=8)
    fit = test.fit(X, Y)
    # summarize scores
    numpy.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X)
    # summarize selected features
    print(features[0:5,:])
