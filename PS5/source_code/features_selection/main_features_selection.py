
import numpy as np
import pandas as pd
#from sklearn import cluster, metrics, ensemble

from sklearn.model_selection import train_test_split
from sklearn.exceptions import ChangedBehaviorWarning
from preprocessing import myPreprocessing
from statistical_analysis import my_statistical_analysis
from feature_univariate import my_feature_univariate
from feature_rfe import my_RFE
from extra_trees_classifier import my_ExtraTreesClassifier
from select_kbest import my_SelectKBest
from heatmap import my_heatmap
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

    print('I. PREPROCESSING')
    processed_data=myPreprocessing(data)
    # show the data
    print("* Print out first samples of processed data for visualiation : ")
    print(processed_data.head())

    print('II. FEATURE SELECTION')
    Y = processed_data['classification']
    X = processed_data.drop(['id','classification'],axis=1)

    # 1.Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
    print('1.Feature univariate')
    my_feature_univariate(X,Y)

    # 2.Feature Extraction with RFE
    print('2.RFE')
    my_RFE(X,Y)

    # 3.Feature Extraction with SelectKBest
    print('3.SelectKBest')
    my_SelectKBest(X,Y)

    # 4. Feature Extraction with ExtraTreesClassifier
    print('4.Extra Trees Classifier')
    my_ExtraTreesClassifier(X,Y)

    # 5.Heatmap
    print("5.Heatmap")
    my_statistical_analysis(processed_data)
    # Heatmap of original data
    my_heatmap(processed_data,"original")
    # Heatmap of selected features data
    selected_data = processed_data[["htn", "dm", "al", "classification"]]
    my_heatmap(selected_data,"selected")
    # print("SAVE THE SELECTED FEATURES DATA INTO csv/selected_data.csv")
    # np.savetxt('csv/selected_data.csv', selected_data, delimiter=',', fmt='%s')
