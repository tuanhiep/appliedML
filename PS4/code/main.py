
import numpy as np
import pandas as pd
#from sklearn import cluster, metrics, ensemble

from sklearn.model_selection import train_test_split
from sklearn.exceptions import ChangedBehaviorWarning
from preprocessing import myPreprocessing
from statistical_analysis import myStatistical_analysis
from Feature_univariate import myFeature_univariate
from Feature_RFE import myFeature_RFE
from PCA import myPCA
from ExtraTreesClassifier import myExtraTreesClassifier
from SelectKBest import mySelectKBest
from Heatmap import myHeatmap
from os import path
import argparse
from warnings import simplefilter
# Supressing Sklearn Future Warnings


simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
simplefilter(action='ignore', category=ChangedBehaviorWarning)
simplefilter(action='ignore', category=UserWarning)


# parse the arguments from command line
parser = argparse.ArgumentParser(description="Kidney Disease Patient Prediction")
parser.add_argument("-data", "--dataSet", type=str, required=True, help="name of the data set for this program")
args = parser.parse_args()

# check if datas et exists
if not path.exists(args.dataSet):
     raise ValueError("Data set not found !")
else:

    data = pd.read_csv(args.dataSet, sep=',')
    # data.columns = ['ID','age', 'bp', 'sg', 'al','su','rbc','pc','pcc', 'ba','bgr','bu','sc','sod','pod','hemo','pcv','wbcc',
    #               'rbcc','htn','dm','cad','appet','pe','ane','class']


    data2=myPreprocessing(data)

    # show the data
    print(data2)

    myStatistical_analysis(data2)


    Y = data2['classification']
    X = data2.drop(['id','classification'],axis=1)


    # Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

    myFeature_univariate(data2,X,Y)

    # Feature Extraction with RFE

    myFeature_RFE(X,Y)

    #Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain.

    # Feature Extraction with PCA

    myPCA(X)

    # Feature Extraction with SelectKBest

    mySelectKBest(X,Y)

    # Feature Extraction with ExtraTreesClassifier

    myExtraTreesClassifier(X,Y)

    #Heatmap original relations

    myHeatmap(data2,"original")
    # drop some not important attributes

    data2 = data2[['pcv', 'htn','dm','al','pc','sc','bgr', 'classification']]

    #Heatmap important relations

    myHeatmap(data2,"important")

    np.savetxt('csv/data.csv', data2, delimiter=',', fmt='%s')
    Y1 = data2['classification']
    X1 = data2.drop(['classification'],axis=1)

    #Split the test set and the traing set

    X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.3, random_state=1)


    #------------------------------------------------------------------#
    print('#------------------------------- To be continued for various types of classifiers -----------------------------------#')
