from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from performance_utils import print_performance_metrics

def my_GaussianNB(X_train,Y_train,X_test,Y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    Y_predTrain = gnb.predict(X_train)
    y_predTest = gnb.predict(X_test)
    print("* Training accuracy ",accuracy_score(Y_train, Y_predTrain))
    print("* Testing accuracy",accuracy_score(Y_test, y_predTest))
    print_performance_metrics(y_predTest,Y_test)
