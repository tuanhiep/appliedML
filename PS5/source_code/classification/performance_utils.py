from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# function to print the performance metrics
def print_performance_metrics(y_predicted, y):
    acc = accuracy_score(y_predicted, y)
    #print("Accuracy = ", acc)
    print('* Precision: %.3f' % precision_score(y_true=y, y_pred=y_predicted, average='micro'))
    print('* Recall: %.3f' % recall_score(y_true=y, y_pred=y_predicted, average='micro'))
    print('* F1: %.3f' % f1_score(y_true=y, y_pred=y_predicted, average='micro'))
