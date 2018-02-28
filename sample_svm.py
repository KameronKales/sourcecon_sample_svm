from sklearn import svm

"""

Above we import the relevant libraries we need to run a basic machine learning algorithm

We are using numpy for data processing

We are using scikit learn to learn the alogirithm.

"""

"""
Here we are loading in the data and printing it out so we can see what we are working on

"""

clf = svm.SVC(gamma=0.001, C=100)

""" Here we are setting clf (classifier) equal to an svm (support vector machine) with .SVC(paramaters)

"""
x = [[1, 1, 1], [2,2,2], [3, 3, 3],
     [4,4,4], [5,5,5], [6,6,6]]

y = [["cat"], ["dog"], ["bird"],
     ["squirrel"], ["dinosaur"], ["pig"]]

clf.fit(x,y)

""" Here we are using our classifier to fit the data using the standard clf.fit(x, y) where x is the data and y is the labels
"""

x_test = [5, 5, 5]
y_test = ["dinosaur"]

model_prediction = clf.predict([x_test])

"""Here we are setting a variable called model_prediction equal to what our algorithm predicted. This can be done by calling .predict and passing the algorithm some value
"""

print "This is our models prediction below"
print model_prediction
