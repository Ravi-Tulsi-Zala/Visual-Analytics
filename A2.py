
# coding: utf-8

# In[1]:


################################################ Libraries ######################################################################

# Pandas package is useful for manipualting data using dataframes , numpy is useful for its powerful feature- arrays
# pyplot calss of matplotlib is usefull for all the visualization stuffs like bar chart and line chart, sklearn contain wide 
# variety of classes and packages such as preprocessing, cross_validation, metrics, linear_model etc. It is also useful for 
# classification algorithms because of its extensive collection of algorithm classes like logisticRegression, DeicisionTreeClassifier

#################################################################################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot 
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors,preprocessing, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

####################### Data load, preprocessing, One Hot Encoding of Categorical Data, Normalization of Numerical data ############################

# In this section of code , I got the count of unique values in capital-gain, capital-loss, and native-country Using value_counts(), 
# 91.2% of capital-gain consists of 0 
# 91.22% of native-country column consists of United States
# 95.29% of capital-loss consists of 0.
# Education number is mapped to Education column do Education column is redundant.
# These columns don't contribute to my salary predictions so I decided to drop these columns  
# Many of the machine learning algorithms can not operate on Categorical Data so I used One Hot encoding to convert the categorical 
# data to numerical data [2].
# Label encoding is not useful in our case as there are a lot of unique categories in a column and it assumes the highe the categorical 
# value, better the category.
# For example, if I have categories A, B, and C labelled with 1,2,and 3 respectively. Lable encoding assumes 3+1/2 = 2 which means 
# Average of A and C is B. This would definitely be a disaster in my salary predictions.
# That is the reason why I chose to apply one hot encoding to categorical data. It binarize the each category and assumes it as a 
# feature to train the model.It uses pandas get_dummies to perform binarization as shown in oneHOtEncoder() method .
# Normalization is required when features of the dataset have different ranges of numerical data [9]. So I used MinMaxScaler of sklearn
# preprocessing to convert the data of numeric features to a common scale (0,1). NormalizeNumeircals method will take numerical column
# and scale it [9].

##########################################################################################################################################



columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","salary"]
dataframe = pd.read_csv('dataset1_processed.csv',names = columns)

def dropRedundantColumns():
    
    dataframe["capital-loss"].value_counts()              #use print to check the output
    dataframe["native-country"].value_counts()
    dataframe["capital-gain"].value_counts()
    dataframe.drop('capital-gain', axis=1, inplace=True)
    dataframe.drop('capital-loss', axis=1, inplace=True)
    dataframe.drop('education', axis=1,inplace=True)
    dataframe.drop('native-country',axis=1,inplace=True)
    
dropRedundantColumns()


def oneHotEncoder(columnName):
    
    newDfTrain = pd.concat([dataframe, pd.get_dummies(dataframe[columnName],prefix=columnName,prefix_sep='_')], axis=1)
    newDfTrain.drop(columnName,axis=1,inplace=True)
    return newDfTrain
      
dataframe = oneHotEncoder("workclass")     
dataframe = oneHotEncoder("marital-status")
dataframe = oneHotEncoder("occupation")    
dataframe = oneHotEncoder("relationship")  
dataframe = oneHotEncoder("race")          
dataframe = oneHotEncoder("sex")
dataframe = oneHotEncoder("education-num")

dataframe['salary'] = dataframe['salary'].apply(lambda salaryData: 1 if salaryData=='>50K' else 0)

def normalizeNumericals(columnName):
    
    dataframe[columnName] = dataframe[columnName].astype(float)
    scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    columnScaled = scale.fit_transform(dataframe[columnName].values.reshape(-1, 1))
    dataframe[columnName] = pd.DataFrame(columnScaled)
    
normalizeNumericals('age')
normalizeNumericals('fnlwgt')
normalizeNumericals('hours-per-week')



################################################ Data Split #####################################################################

# For splitting the data into train and validation sets , I used cross validation method of sklearn. It will split the dataset as
# 70% train data and 30% validation data as I set the test size to 0.3 [8].
# y variables will be my target data and X variables will be my independent variables.
# labels are useful in labelling confusion matrix.

##################################################################################################################################



data = np.array(dataframe.drop(['salary'], 1))
target = np.array(dataframe['salary']) 

X_train, X_val, y_train, y_val = cross_validation.train_test_split(data, target, test_size=0.3)
labels = [0, 1]




####################### Training of Logistic Regression, K-Nearest Neighbors, and Decision Tree algorithms ############################

# I used Logistic Regression, K-Nearest Neighbors, and Decision Tree algorithms for training. 
# Logistic regression is a parametric model whereas KNN and Deicision tree are Non Parametric Model. I implemented classifyDatasets(),
# calculateAverageClassAccuracy(),and compareResults().
# classifyDatasets() takes classifier as argument and it will use fit() method of sklearn to train the ML model and score() function
# to find the acuuracy of classifier [8].
# Logistic regression takes hyperparameters like penalty, dual, and iterations, K-nearest neighbours takes n_neighbors hyperparameter
# and decision tree takes max_depth,criterion of entroy or gini index as hyperparameter in our case. Hyperparameter are useful in finding the best accuracy scores for 
# a classifier [7].
# predict() will perform the predictions and confusion_matrix will be useful for finding the true positive rate, true negative rate,
# average class accuracy for both positive and negative class
# From the confusion matrix statisctics like TP(True Positives), FP(False Positives), TN(True Negatives), FN(False Negatives) will
# be available [6]. This information is useful in finding the per class accuracy and average class accuracy as shown in calculateAverageClassAccuracy()
# compareResults() will then compare the accuracy scores for each algorithms used and visualizing using bar chart [4]. From the accuracy score and average accuracy scores
# it is clearly seen that, K-nearest neighbors classifier has accuracy score of around 89% in both training sets and validation sets but 
# While considering average class accuracy KNN shows 10% of difference between avg class accuracies of Train set(74%) and Validation sets(84%).
# Using Logistic Regression classifier, accuracy scores for train and validation sets are of around 83% but average accuracy scores for both the sets
# reached to 74%.
# Using Decision Tree Classifier, accuracy scores for both the sets are around 84% but average accuracy scores shows 4% difference

############################################################################################################################################



def classifyDatasets(classifier,classifierName):
    
    classifierTrainset = classifier.fit(X_train, y_train)
    accTrain = classifierTrainset.score(X_train, y_train) * 100
    print('\t \t Accuracy of '+ classifierName +' on training set: ' + str(accTrain))
    classifierValidationset = classifier.fit(X_val, y_val)
    accVal = classifierValidationset.score(X_val, y_val) * 100
    print('\t \t Accuracy of '+ classifierName +' on validation set: ' + str(accVal),'\n')
    y_train_pred = classifierTrainset.predict(X_train)
    cm_train = confusion_matrix(y_train, y_train_pred, labels)
    y_val_pred = classifierValidationset.predict(X_val)
    cm_val = confusion_matrix(y_val, y_val_pred, labels)
    
    return cm_train,cm_val,accTrain,accVal

def calculateAverageClassAccuracy(cm_train,cm_val):
    
    print('\t Trainset Statistics  : \n')
    
    TP =  cm_train[0][0]
    FP =  cm_train[0][1]
    FN =  cm_train[1][0]
    TN =  cm_train[1][1]
    
    print('\t \t True Positives in train set  :', TP)
    print('\t \t False Positives in train set :', FP)
    print('\t \t False Negatives in train set :', FN)
    print('\t \t True Negatives in train set  :', TN)
    
    positiveAccuracy = TP*100 / (TP+FP)
    negativeAccuracy = TN*100 / (TN+FN)
    averageClassAccuracyTrain = (positiveAccuracy + negativeAccuracy) / 2
    overallAccuracy = (TP + TN ) * 100 / (TP + FP + TN + FN)
    print('\t \t Average Class Accuracy in Train Set : ', averageClassAccuracyTrain) 
    print('\n \n')
    
    print('\t Validation Set Statistics  : \n')
    
    TP =  cm_val[0][0]
    FP =  cm_val[0][1]
    FN =  cm_val[1][0]
    TN =  cm_val[1][1]
    
    print('\t \t True Positives in validation set  :', TP)
    print('\t \t False Positives in validation set :', FP)
    print('\t \t False Negatives in validation set :', FN)
    print('\t \t True Negatives in validation set  :', TN)
        
    positiveAccuracy = TP*100 / (TP+FP)
    negativeAccuracy = TN*100 / (TN+FN)
    averageClassAccuracyVal = (positiveAccuracy + negativeAccuracy) / 2
    overallAccuracy = (TP + TN ) * 100 / (TP + FP + TN + FN)
    print('\t \t Average Class Accuracy in Validation Set : ', averageClassAccuracyVal) 
    print('\n \n')
    
    return averageClassAccuracyTrain,averageClassAccuracyVal
    

def compareResults(classifiers,accuracies, setType, scoreName):
    
    index = np.arange(len(classifiers))
    plot.bar(index, accuracies)
    plot.xlabel('Classifiers', fontsize=12)
    plot.ylabel(scoreName + ' Score for '+ setType +' Set', fontsize=12)
    plot.xticks(index, classifiers, fontsize=10, rotation=30)
    plot.title(scoreName + ' Score Comparison of Classifiers on '+setType+' Set')
    plot.show()
    
# Logistic Regression classsifier   
print('Logistic Regression Classifier : \n')    
logisticRegressionClf = LogisticRegression(penalty='l2',dual=False,max_iter=100)
cm_train,cm_val,accTrainLog,accValLog = classifyDatasets(classifier=logisticRegressionClf,classifierName="Logistic Regression Classifier ")
averageClassAccuracyTrainLog,averageClassAccuracyValLog = calculateAverageClassAccuracy(cm_train,cm_val) 


# Decision Tree Classifier
print('Decision Tree Classifier : \n')
decisionTreeClf = DecisionTreeClassifier(criterion='entropy', max_depth=8)
cm_train,cm_val,accTrainDec ,accValDec = classifyDatasets(classifier=decisionTreeClf,classifierName="Decision Tree Classifier")
averageClassAccuracyTrainDec,averageClassAccuracyValDec=calculateAverageClassAccuracy(cm_train,cm_val)  

#  KNN classifier
print('KNN Classifier : \n')
knnClf = neighbors.KNeighborsClassifier(n_neighbors=3)
cm_train,cm_val,accTrainKnn,accValKnn = classifyDatasets(classifier=knnClf,classifierName="K-Nearest Neighbors classifier")
averageClassAccuracyTrainKnn,averageClassAccuracyValKnn = calculateAverageClassAccuracy(cm_train,cm_val)  

# Comparing the results of classifier on train and validation sets
classifiersList = ["Logistic Regression", "Decision Tree","KNN Classifier"]
trainAccuracies = [accTrainLog,accTrainDec,accTrainKnn]
valAccuracies = [accValLog,accValDec,accValKnn]
trainAvgClsAcc = [averageClassAccuracyTrainLog,averageClassAccuracyTrainDec,averageClassAccuracyTrainKnn]
valAvgClsAcc = [averageClassAccuracyValLog,averageClassAccuracyValDec,averageClassAccuracyValKnn]

compareResults(classifiers=classifiersList,accuracies=trainAccuracies, setType="Train",scoreName="Accuracy")
compareResults(classifiers=classifiersList,accuracies=valAccuracies, setType="Validation",scoreName="Accuracy")
compareResults(classifiers=classifiersList,accuracies=trainAvgClsAcc, setType="Train",scoreName="Avg Accuracy")
compareResults(classifiers=classifiersList,accuracies=valAvgClsAcc, setType="Validation",scoreName="Avg Accuracy")



############# Variation in validation set avg class accuracy according to maxdepth parameter of decision tree ############################

# testingParameter() will check the average accuracy scores and accuracy scores using variation of max depth hyperparameter of
# decision tree classifier in validation sets [1] [3].
# It will calculate average class accuracy scores, overall accuracy scores for different values of max_depth and append to the 
# respective lists.
# Line chart will take max_depth and accuracy scores as x and y labels respectively.Line chart will be useful in visualizing the 
# variation of max_depth parameter (Accuracy vs max_depth) [4] [5]

#######################################################################################################################################

avgAcc  = []
depths = []
accuracyKnn = []

def testingParameter():
    
    for i in range(1,11):

        decisionTreeClf = DecisionTreeClassifier(criterion='entropy', max_depth=i) 
        decisionTreeClfVal = decisionTreeClf.fit(X_val, y_val)
        accVal = decisionTreeClfVal.score(X_val, y_val) * 100
        y_val_pred = decisionTreeClfVal.predict(X_val)
        cm_val = confusion_matrix(y_val, y_val_pred, labels)

        TP =  cm_val[0][0]
        FP =  cm_val[0][1]
        FN =  cm_val[1][0]
        TN =  cm_val[1][1]

        positiveAccuracy = TP*100 / (TP+FP)
        negativeAccuracy = TN*100 / (TN+FN)
        averageClassAccuracyVal = (positiveAccuracy + negativeAccuracy) / 2

        depths.append(i)
        avgAcc.append(averageClassAccuracyVal)
        accuracyKnn.append(accVal)
        
testingParameter()    

def plotAvgclassAcc(depths,avgAcc):
    
    plot.plot(depths,avgAcc, color='g')
    plot.xlabel('Maximum Depth ')
    plot.ylabel('Average Class Accuracy')
    plot.title('Decision Tree Classifier Variation in Avg Class Accuracy')
    plot.show()

def plotoverallAcc(depths,accuracyKnn):
    
    plot.plot(depths,accuracyKnn, color='g')
    plot.xlabel('Maximum Depth ')
    plot.ylabel('Accuracy')
    plot.title('Decision Tree Classifier Variation in Accuracy')
    plot.show()
    
    
plotAvgclassAcc(depths,avgAcc)
plotoverallAcc(depths,accuracyKnn)

######################################## Parameter Tuning of Decision Tree Classifier ############################################

# Reason behind choosing Decision Tree Algorithm is the fitting and predictions are faster than K-nearest neightbors and thr accuracy
# scores are better than Logistic Regression algorithm. Moreover, K-nearest neighbors is lazy learner which take less time in 
# training but more time predicting whereas Decision Tree Algorithm which is eager learner hence it is faster in both the case [3].
# Our ultimate goals is salary predication which makes Decision Tree Algorithm a better choice in our case. Decision tree is also
# useful to pick up nonlinearities in data which makes it fairly accurate than others [3].
# Criterion hyperparameter of decision tree can have Entropy and Gini index which are considered for the tuning in our case [3].
# After visualizing the data and doing some research , Gini index takes a little less time in computation than entropy criterion
# We are dealing with a lot of data ,so I decided to go with Gini Index.Using Gini index and setting max_depth to 9 helped me in
# increasing the average class accuracy and accuracy scores to 86% and 76% in validations sets.Which are 3% more than I achieved
# using entropy criterion and max_depth of 7.
# decTreeUsingGini() calculates the accuracy scores for validation sets for different hyperparameter value (criterion='gini', max_depth=9).

#######################################################################################################################################

def decTreeUsingGini():
    
    decisionTreeClf = DecisionTreeClassifier(criterion='gini', max_depth=9)
    decisionTreeClfVal = decisionTreeClf.fit(X_val, y_val)
    accVal = decisionTreeClfVal.score(X_val, y_val) * 100
    print('After Hyperparameter Tuning in Decision Tree : \n ')
    print('\t \t Accuracy of Decision Tree Classifier on validation set using gini index: ' + str(accVal),'\n')
    y_val_pred = decisionTreeClfVal.predict(X_val)
    cm_val = confusion_matrix(y_val, y_val_pred, labels)
    
    TP =  cm_val[0][0]
    FP =  cm_val[0][1]
    FN =  cm_val[1][0]
    TN =  cm_val[1][1]
 
    positiveAccuracy = TP*100 / (TP+FP)
    negativeAccuracy = TN*100 / (TN+FN)
    averageClassAccuracyVal = (positiveAccuracy + negativeAccuracy) / 2
    print('\t \t Average Class Accuracy of Decision Tree Classifier on validation set using gini index: ' + str(averageClassAccuracyVal),'\n')
        
decTreeUsingGini()

###################################################### Predictions ###############################################################

# After tuning the classifier, it is time to do the predictions. First I converted dataset1_test.csv  into dataframe and then
# I performed the normalization, one hot encoding to test dataframe. Then I converted it into numpy array start predicting the 
# salary for the test dataset usiing Decision Tree Classifier. It will append the results to B00805073_prediction.csv

#################################################################################################################################### 

columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"]
dataframe_test = pd.read_csv('dataset1_test.csv',names = columns );

dataframe_test.drop('capital-gain', axis=1, inplace=True)
dataframe_test.drop('capital-loss', axis=1, inplace=True)
dataframe_test.drop('education', axis=1,inplace=True)
dataframe_test.drop('native-country',axis=1,inplace=True)

def oneHotEncoderTest(columnName):
    
    newDfTest = pd.concat([dataframe_test, pd.get_dummies(dataframe_test[columnName],prefix=columnName,prefix_sep='_')], axis=1)
    newDfTest.drop(columnName,axis=1,inplace=True)
    return newDfTest

dataframe_test = oneHotEncoderTest("workclass")     
dataframe_test = oneHotEncoderTest("marital-status")
dataframe_test = oneHotEncoderTest("occupation")    
dataframe_test = oneHotEncoderTest("relationship")  
dataframe_test = oneHotEncoderTest("race")          
dataframe_test = oneHotEncoderTest("sex")           
dataframe_test = oneHotEncoderTest("education-num") 

def normalizeNumericalsTest(columnName):
    
    dataframe_test[columnName] = dataframe_test[columnName].astype(float)
    scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    columnScaled = scale.fit_transform(dataframe_test[columnName].values.reshape(-1, 1))
    dataframe_test[columnName] = pd.DataFrame(columnScaled)
    
normalizeNumericalsTest('age')
normalizeNumericalsTest('fnlwgt')
normalizeNumericalsTest('hours-per-week')

X_test = np.array(dataframe_test)
y_test_pred =  decisionTreeClf.predict(X_test)
predictions = y_test_pred.tolist()

def exportPredictions():
    count = 0     
    for i in range(len(predictions)):

        if(predictions[i]==0):
            count+=1
            predictions[i] = '<=50K'

        elif(predictions[i]==1):
            predictions[i] = '>50K'
    
    print(count)
    
    df = pd.DataFrame(predictions)        
    df.to_csv('B00805073_prediction.csv', header=False,index=False)

exportPredictions()

##################################################### References ########################################################################


# [1]"Decision Trees: How to Optimize My Decision-Making Process?", Medium, 2019. [Online]. Available: https://medium.com/cracking-the-data-science-interview/decision-trees-how-to-optimize-my-decision-making-process-e1f327999c7a. [Accessed: 07- Jun- 2019].

# [2]J. Brownlee, "Why One-Hot Encode Data in Machine Learning?", Machine Learning Mastery, 2019. [Online]. Available: https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/. [Accessed: 07- Jun- 2019].

# [3]"InDepth: Parameter tuning for Decision Tree", Medium, 2019. [Online]. Available: https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3. [Accessed: 07- Jun- 2019].

# [4]"Data Visualization in Python — Line Graph in Matplotlib", Medium, 2019. [Online]. Available: https://medium.com/@pknerd/data-visualization-in-python-line-graph-in-matplotlib-9dfd0016d180. [Accessed: 07- Jun- 2019].

# [5]2019. [Online]. Available: https://medium.com/python-pandemonium/data-visualization-in-python-bar-graph-in-matplotlib-f1738602e9c4. [Accessed: 07- Jun- 2019].

# [6]D. Gopinath, "Confusion Matrix - Get Items FP/FN/TP/TN - Python", Data Science Stack Exchange, 2019. [Online]. Available: https://datascience.stackexchange.com/questions/28493/confusion-matrix-get-items-fp-fn-tp-tn-python. [Accessed: 07- Jun- 2019].

# [7]A. Zheng, "Evaluating Machine Learning Models", O'Reilly Media, 2019. [Online]. Available: https://www.oreilly.com/ideas/evaluating-machine-learning-models/page/3/evaluation-metrics. [Accessed: 07- Jun- 2019].

# [8]"Solving A Simple Classification Problem with Python — Fruits Lovers’ Edition", Towards Data Science, 2019. [Online]. Available: https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2. [Accessed: 07- Jun- 2019].

# [9]M. Aquilina and B. Musa, "Normalize columns of pandas data frame", Stack Overflow, 2019. [Online]. Available: https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame. [Accessed: 07- Jun- 2019].

