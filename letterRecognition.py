#!/usr/bin/python

import numpy
import pandas
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize


class Visualisation:
	
	def __init__(self, classes, XTrain=None, yTrain=None, XTest=None, yTest=None):
		self.classes = classes
		self.XTrain = XTrain
		self.yTrain = yTrain
		self.XTest = XTest
		self.yTest = yTest

	def getPCA(self, X):
		pca = PCA(n_components=2)
		pca.fit(X)
		return pca.transform(X)

	def showPCAGraph(self, X, y, title):
		reduction = self.getPCA(X)
		plt.figure()
		plt.title(title)
		plt.scatter(reduction[:, 0], reduction[:, 1], c=y, cmap=plt.cm.coolwarm)
		plt.show()

	def adjustY(self, classifier):
		yTest = numpy.array(self.yTest)
		yTestSize = len(yTest)
		yTestMatrix = numpy.zeros((yTestSize, self.classes))
		for i in range(yTestSize):
			yTestMatrix[(i, yTest[i])] = 1
		classes = [x for x in range(self.classes)]
		yTestMatrix = label_binarize(yTestMatrix, classes=classes)
		oneVsRestClassifier = OneVsRestClassifier(classifier)
		yTestScore = oneVsRestClassifier.fit(self.XTrain, self.yTrain).decision_function(self.XTest)
		return yTestMatrix, yTestScore

	def getRatesAndCurve(self, yTestMatrix, yTestScore):
		falsePositiveRate = dict()
		truePositiveRate = dict()
		areaUnderCurve = dict()
		for i in range(self.classes):
			falsePositiveRate[i], truePositiveRate[i], _ = roc_curve(yTestMatrix[:, i], yTestScore[:, i])
			areaUnderCurve[i] = auc(falsePositiveRate[i], truePositiveRate[i])
		return falsePositiveRate, truePositiveRate, areaUnderCurve

	def showROCGraph(self, classifier, titleMethod, letter=None):
		yTestMatrix, yTestScore = self.adjustY(classifier)
		falsePositiveRate, truePositiveRate, areaUnderCurve = self.getRatesAndCurve(yTestMatrix, yTestScore)
		title = titleMethod + " ROC "
		lw = 2
		plt.figure()
		if letter == None:
			title += "for all classes"
			for i in range(self.classes):
				plt.plot(falsePositiveRate[i], truePositiveRate[i], lw=lw, label="ROC curve of class {0} (area = {1:0.6f})".format(chr(i + ord('A')), areaUnderCurve[i]))
		else:
			index = ord(letter) - ord('A')
			if index < 0 or index >= self.classes:
				print("Wrong parameter!")
				return
			title += "for class " + letter
			plt.plot(falsePositiveRate[index], truePositiveRate[index], lw=lw, label="ROC curve of class {0} (area = {1:0.6f})".format(chr(index + ord('A')), areaUnderCurve[index]))
		plt.plot([0, 1], [0, 1], "k--", lw=lw)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel("False Positive Rate")
		plt.ylabel("True Positive Rate")
		plt.title(title)
		plt.legend(loc="lower right")
		plt.show()


class LetterRecognition:

	def __init__(self, data, split=0.75):
		self.dataFrame = pandas.read_csv(data, sep=",", header=None)
		self.splitData(split)
		self.divideData()
		self.classes = 26

	def splitData(self, split):
		ratio = int(len(self.dataFrame) * split)
		self.trainSet = self.dataFrame[:ratio]  
		self.testSet = self.dataFrame[ratio:]

	def divideData(self):
		self.XTrain = self.trainSet.ix[:, 1:]
		self.yTrain = self.trainSet.ix[:, 0].apply(lambda x: ord(x) - ord('A'))
		self.XTest = self.testSet.ix[:, 1:]
		self.yTest = self.testSet.ix[:, 0].apply(lambda x: ord(x) - ord('A'))

	def showPCATrainGraph(self):
		visualisation = Visualisation(self.classes)
		visualisation.showPCAGraph(self.XTrain, self.yTrain, "Train Data")

	def showPCATestGraph(self):
		visualisation = Visualisation(self.classes)
		visualisation.showPCAGraph(self.XTest, self.yTest, "Test Data")

	def supportVectorClassifier(self):
		model = svm.SVC(C=50.0, kernel="rbf")
		model.fit(self.XTrain, self.yTrain)
		print("SVM: Score = " + str(self.getScore(model)))
		print("SVM: Cross Score = " + str(self.getCrossScore(model)))
		return model

	def randomForestClassifier(self):
		model = RandomForestClassifier(n_estimators=200, max_depth=500)
		model.fit(self.XTrain, self.yTrain)
		print("RFC: Score = " + str(self.getScore(model)))
		print("RFC: Cross Score = " + str(self.getCrossScore(model)))
		return model

	def decisionTreeClassifier(self):
		model = tree.DecisionTreeClassifier()
		model.fit(self.XTrain, self.yTrain)
		print("DTC: Score = " + str(self.getScore(model)))
		print("DTC: Cross Score = " + str(self.getCrossScore(model)))
		return model

	def logisticRegression(self):
		model = LogisticRegression(C=20.0)
		model.fit(self.XTrain, self.yTrain)
		print("Logistic regression: Score = " + str(self.getScore(model)))
		print("Logistic regression: Cross Score = " + str(self.getCrossScore(model)))
		return model

	def getScore(self, model):
		return model.score(self.XTest, self.yTest)

	def getCrossScore(self, model):
		return numpy.mean(cross_val_score(model, self.XTest, self.yTest, cv=5))

	def showROCGraph(self, classifier, title, letter=None):
		visualisation = Visualisation(self.classes, self.XTrain, self.yTrain, self.XTest, self.yTest)
		visualisation.showROCGraph(classifier, title, letter)


letterRecognition = LetterRecognition("files/data.csv", 0.75)
letterRecognition.showPCATrainGraph()
letterRecognition.showPCATestGraph()
svc = letterRecognition.supportVectorClassifier()
rfc = letterRecognition.randomForestClassifier()
dtc = letterRecognition.decisionTreeClassifier()
logisticRegression = letterRecognition.logisticRegression()
letterRecognition.showROCGraph(svc, "SVC")
letterRecognition.showROCGraph(logisticRegression, "Logistic Regression")






