from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from performance_utils import print_performance_metrics

def my_KNeighborsClassifier(X_train_std,X_test,y_train,y_test):
   acc_list_train=[]
   acc_list_test=[]
   n=[i for i in range(1,51)]
   for i in range(1,51):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train_std, y_train)
        acc_list_train.append(accuracy_score(y_train,knn.predict(X_train_std)))
        acc_list_test.append(accuracy_score(y_test,knn.predict(X_test)))


   fig,axes_knn=plt.subplots(1,1)
   plt.plot(n,acc_list_train,color='Red',label='Training accuracy')
   plt.plot(n,acc_list_test,color='Blue',label='Testing accuracy')
   plt.ylabel('Accuracy Score')
   plt.xlabel('n (Nearest Neighbors)')
   plt.title('KNN')
   plt.legend(loc='upper right')
   print("* Save KNN classification result into image/knn.png")
   plt.savefig('image/knn.png')

   max_indexes_test=[i+1 for i,value in enumerate(acc_list_test) if value == max(acc_list_test)]
   print("* Best K is "+ str(max_indexes_test[0]) )
   clf = KNeighborsClassifier(n_neighbors=max_indexes_test[0], metric='euclidean', p=2)
   clf.fit(X_train_std, y_train)
   Y_predTrain = clf.predict(X_train_std)
   Y_predTest_best = clf.predict(X_test)
   print("* Training accuracy : ",accuracy_score(y_train, Y_predTrain))
   print("* Testing accuracy : ",accuracy_score(y_test, Y_predTest_best))
   print_performance_metrics(Y_predTest_best,y_test)
