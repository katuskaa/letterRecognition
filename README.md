# Letter Recognition
The aim of this project is recognition of english capital letters. Project implements four methods of multiclassification.

Usage of class LetterRecognition:

<code>
letterRecognition = LetterRecognition("files/data.csv", 0.75)
<code>

First parameter is file containing input data. Second parameter is optional, determines ratio between training and testing data set, default is 0.75.

<code>
letterRecognition.showPCATrainGraph()
<code>

Using PCA method to reduce dimensions. Shows graph displaying two main compomnents of training set.

<code>
letterRecognition.showPCATestGraph()
<code>

Using PCA method to reduce dimensions. Shows graph displaying two main compomnents of testing set.
 
<code>
svc = letterRecognition.supportVectorClassifier()
<code>

Returns trained Support Vector Classifier.

<code>
rfc = letterRecognition.randomForestClassifier()
<code>

Returns trained Random Forest Classifier.

<code>
dtc = letterRecognition.decisionTreeClassifier()
<code>

Returns trained Decision Tree Classifier.

<code>
logisticRegression = letterRecognition.logisticRegression()
<code>

Returns trained Logistic Regression.

<code>
letterRecognition.showROCGraph(svc, "SVC")
<code>

Shows graph of all ROC classes in one graph for Support Vector Classifier.

<code>
letterRecognition.showROCGraph(logisticRegression, "Logistic Regression")
<code>

Shows graph of all ROC classes in one graph for Logistic Regression.
