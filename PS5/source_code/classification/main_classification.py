
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
#from sklearn import cluster, metrics, ensemble

from sklearn.model_selection import train_test_split
from sklearn.exceptions import ChangedBehaviorWarning
from sklearn import preprocessing
from adaboost import my_AdaBoost
from bagging import my_Bagging
from random_forest import my_RandomForestClassifier
from gaussian_nb import my_GaussianNB
from knn import my_KNeighborsClassifier
from decision_tree import my_DecisionTree
from logistic_regression import my_LogisticRegression
from svm import my_SVM
from non_linear_svm import my_NonLinearSVM

# Supressing Sklearn Future Warnings

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
simplefilter(action='ignore', category=ChangedBehaviorWarning)
simplefilter(action='ignore', category=UserWarning)


data = pd.read_csv('csv/kidneyclean.csv')
data.columns = ['id','age', 'bp', 'sg', 'al','su','rbc','pc','pcc', 'ba','bgr','bu','sc','sod','pod','hemo','pcv','wbcc',
              'rbcc','htn','dm','cad','appet','pe','ane','classification']

#best_attributes = ["pcv", "htn", "dm", "al", "pc", "sc", "bgr", "classification"]
#best_attributes = [ "htn", "dm", "al", "sc", "bgr", "classification"]
#best_attributes = [ "htn", "dm", "al", "bgr", "classification"]
best_attributes = [ "htn", "dm", "al", "classification"]
drop_list = [i for i in data.columns if i not in best_attributes]
data.drop(drop_list, axis=1, inplace=True)

# separate the data from the target attributes
Y = data['classification']
X = data.drop(['classification'], axis=1)

# Create training set and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1,shuffle=True)
 # Save training set and testing set into files
np.savetxt('csv/X_train.csv', X_train, delimiter=', ')
np.savetxt('csv/X_test.csv', X_test, delimiter=', ')
np.savetxt('csv/Y_train.csv', Y_train, delimiter=', ')
np.savetxt('csv/Y_test.csv', Y_test, delimiter=', ')

print("III. MACHINE LEARNING CLASSIFICATION ALGORITHMS")

#Gaussian Classifier
print('1.Gaussian NB')
start=time.time()
my_GaussianNB(X_train,Y_train,X_test,Y_test)
stop=time.time()
print("* Runtime for model Gaussian NB  is = "+str(stop-start)+"s")

#KNN
print('2.KNN ')
start= time.time()
my_KNeighborsClassifier(X_train,X_test,Y_train,Y_test)
stop = time.time()
# Because we run KNN with 51 different parameter K
print("* Runtime for model KNN  is = "+str((stop-start)/51)+"s")

# Decision Tree with Entropy and Gini
print('3.Decision Tree')
start= time.time()
my_DecisionTree(X_train,X_test,Y_train,Y_test)
stop= time.time()
# Because we run Decision Tree for 4 different parameter max depth
print("* Runtime for model Decision Tree is = "+str((stop-start)/4)+"s")

## Logistic Regression
print('4.Logistic Regression')
start= time.time()
my_LogisticRegression(X_train,Y_train,X_test,Y_test)
stop= time.time()
# Because we run Logistic Regression for 4 different parameter C
print("* Runtime for model Logistic Regression is = "+str((stop-start)/4)+"s")

# SVM
print('5.SVM')
start= time.time()
my_SVM(X_train,Y_train,X_test,Y_test)
stop= time.time()
# Because we run Logistic Regression for 10 different parameter C
print("* Runtime for model SVM is = "+str((stop-start)/10)+"s")

# Non Linear SVM
print('6.Non Linear SVM')
start= time.time()
my_NonLinearSVM(X_train,Y_train,X_test,Y_test)
stop= time.time()
# Because we run Logistic Regression for 10 different parameter C
print("* Runtime for model Non Linear SVM is = "+str((stop-start)/10)+"s")

# Ensemble Methods
num_base_classifiers = 500
max_depth_EM = 10
train_Acc = []
test_Acc = []

# Random Forest
print('7.Random Forest')
start= time.time()
my_RandomForestClassifier(X_train,X_test,Y_train,Y_test,num_base_classifiers,train_Acc, test_Acc)
stop= time.time()
print("* Runtime for model Random Forest is = "+str(stop-start)+"s")

# Bagging
print('8.Bagging')
start= time.time()
my_Bagging(X_train,X_test,Y_train,Y_test,num_base_classifiers, max_depth_EM,train_Acc, test_Acc)
stop= time.time()
print("* Runtime for model Bagging is = "+str(stop-start)+"s")

# Adaboost
print('9.Adaboost')
start= time.time()
my_AdaBoost(X_train,X_test,Y_train,Y_test,num_base_classifiers, max_depth_EM,train_Acc, test_Acc)
stop= time.time()
print("* Runtime for model Adaboost is = "+str(stop-start)+"s")

methods = ['Random Forest', 'Bagging', 'AdaBoost']
plt.plot(methods, train_Acc, 'ro-')
plt.plot(methods, test_Acc, 'bv--')
plt.xlabel("Methods")
plt.ylabel("Accuracy")
plt.title("Ensemble Method")
print("*** Save Ensemble Method : Random Forest, Bagging, Adaboost classification result into image/ensemble_method.png")
plt.savefig("image/ensemble_method.png")
plt.show()
