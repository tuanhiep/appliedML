import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def my_SelectKBest(X,Y):
    
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']
    print("* Print out 10 best features")
    print(featureScores.nlargest(10,'Score'))
