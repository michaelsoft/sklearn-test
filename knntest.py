from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target


X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size = 0.2 )
knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)
print(knn.predict(X_test))
print(Y_test) 
