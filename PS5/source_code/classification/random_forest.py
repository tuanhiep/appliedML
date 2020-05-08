from sklearn.metrics import accuracy_score
from sklearn import ensemble
from performance_utils import print_performance_metrics

def my_RandomForestClassifier(X_train,X_test,Y_train,Y_test,numBaseClassifiers,train_Acc, test_Acc):
    clf = ensemble.RandomForestClassifier(n_estimators=numBaseClassifiers)
    clf.fit(X_train, Y_train)
    Y_predict_train_EM = clf.predict(X_train)
    Y_predict_test_EM = clf.predict(X_test)
    train_Acc.append(accuracy_score(Y_train, Y_predict_train_EM))
    test_Acc.append(accuracy_score(Y_test, Y_predict_test_EM))
    print("* Train accuracy = %f and Test accuracy = %f " % (
    accuracy_score(Y_train, Y_predict_train_EM), accuracy_score(Y_test, Y_predict_test_EM)))
    print_performance_metrics(Y_predict_test_EM,Y_test)
