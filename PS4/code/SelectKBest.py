import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def mySelectKBest(X,Y):
    print('#----------------------------- Feature Extraction : SelectKBest -------------------------------------#')
    
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']
    print(featureScores.nlargest(10,'Score'))