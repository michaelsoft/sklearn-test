from sklearn import linear_model


x=[[0,0],[0,1]]
y=[0,1]
model = linear_model.LinearRegression()
model.fit(x,y)
test_x=[[0,0.1]]
result = model.predict(test_x)
print(result)
