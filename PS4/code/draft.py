
import numpy
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn import cluster, metrics, ensemble
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score


data = pd.read_csv('csv/kidney.csv',header=None, sep=',')
data.columns = ['age', 'bp', 'sg', 'al','su','rbc','pc','pcc', 'ba','bgr','bu','sc','sod','pod','hemo','pcv','wbcc',
               'rbcc','htn','dm','cad','appet','pe','ane','class']

# The original data has some mistype value, replace
data.replace('ckd\t', 'ckd', regex=True)
# Make sure that there are only 2
print(data['class'].value_counts())
# Replace missing data
data = data.replace('?',np.NaN)

print('Number of instances = %d' % (data.shape[0]))
print('Number of attributes = %d' % (data.shape[1]))

print('Number of missing values:')
for col in data.columns:
    print('\t%s: %d' % (col,data[col].isna().sum()))

# Drop the missing data
print('Number of rows in original data = %d' % (data.shape[0]))
data2 = data.dropna()
print('Number of rows after discarding missing values = %d' % (data2.shape[0]))

# Make the text data become numeric
data2 = data2.replace('normal',1)
data2 = data2.replace('abnormal',0)
data2 = data2.replace('present',1)
data2 = data2.replace('notpresent',0)
data2 = data2.replace('no',0)
data2 = data2.replace('yes',1)
data2 = data2.replace('poor',0)
data2 = data2.replace('good',1)
data2 = data2.replace('notckd',0)
data2 = data2.replace('ckd',1)


# Save data to a csv file
np.savetxt('csv/kidneyclean.csv', data2, delimiter=',', fmt='%s')
# show the data
print(data2)
# Statistical analysis
for col in data2.columns:
    if is_numeric_dtype(data2[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % data2[col].mean())
        print('\t Standard deviation = %.2f' % data2[col].std())
        print('\t Minimum = %.2f' % data2[col].min())
        print('\t Maximum = %.2f' % data2[col].max())
#Data Covariance:
data2.cov()
#Data Correlation:
data2.corr()




# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

Y = data2['class']
X = data2.drop(['class'],axis=1)
# feature extraction
test = SelectKBest(score_func=chi2, k=8)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])



# Feature Extraction with RFE


#Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain.

model = LogisticRegression()
rfe = RFE(model, 8)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

# Feature Extraction with PCA


pca = PCA(n_components=8)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)


# Feature Extraction with SelectKBest
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
print(featureScores.nlargest(10,'Score'))
# Feature Extraction with ExtraTreesClassifier


model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Importance Scores')
# naming the y axis
plt.ylabel('Attributes')

# giving a title to my graph
plt.title('Feature Importance')
plt.savefig("image/featureimportance.png")
plt.clf()

#Heatmap relations
corrmat = data2.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data2[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.savefig("image/corelations.png")
plt.clf()


# drop some not important attributes
data2 = data2[['pcv', 'htn','dm','al','pc','su','bgr', 'class']]
np.savetxt('csv/data.csv', data2, delimiter=',', fmt='%s')
Y1 = data2['class']
X1 = data2.drop(['class'],axis=1)
#Split the test set and the traing set
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.3, random_state=1)

#Create a Gaussian Classifier
gnb = GaussianNB()
#Train the model using the training sets
gnb.fit(X_train, Y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
print("Accuracy - GaussianNB: ",metrics.accuracy_score(Y_test, y_pred))

# Decision Tree

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf = clf.fit(X_train, Y_train)
Y_predTrain = clf.predict(X_train)
Y_predTest = clf.predict(X_test)
print("Accuracy - Decision Tree: ",metrics.accuracy_score(Y_test, Y_predTest))

# KNN

clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean', p=2)
clf.fit(X_train, Y_train)
Y_predTrain = clf.predict(X_train)
Y_predTest1 = clf.predict(X_test)
print("Accuracy - KNeighborsClassifier: ",metrics.accuracy_score(Y_test, Y_predTest1))
