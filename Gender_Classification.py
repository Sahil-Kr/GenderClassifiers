from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

#defining the classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_KNN = KNeighborsClassifier()
clf_NB = GaussianNB()

#dataset
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


#training of data
clf_tree = clf_tree.fit(X, Y)
clf_svm = clf_svm.fit(X,Y)
clf_KNN = clf_KNN.fit(X,Y)
clf_NB = clf_NB.fit(X,Y)

#testing of data
#decision Tree
prediction_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, prediction_tree)*100
print('Accuracy of Decision Tree classifier is:{}'.format(acc_tree))

#SVM
prediction_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, prediction_svm )*100
print('Accuracy of SVM classifier is:{}'.format(acc_svm))

#KNN
prediction_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, prediction_KNN)*100
print('Accuracy of KNN is:{}'.format(acc_KNN))

#Naive Bayes
prediction_NB = clf_NB.predict(X)
acc_NB = accuracy_score(Y, prediction_NB)*100
print('Accuracy of Naive Bayes is:{}'.format(acc_NB))

#displaying the result 
print(prediction_tree)
print(prediction_svm)
print(prediction_KNN)
print(prediction_NB)

# The best classifier from svm, tree, KNN, NaiveBayes
index = np.argmax([acc_tree, acc_svm, acc_KNN, acc_NB])
classifiers = {0: 'Tree', 1: 'SVM', 2: 'KNN', 3:'NaiveBayes'}
print('Best gender classifier is {}'.format(classifiers[index]))