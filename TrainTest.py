import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Dataset illustrates 100 customers in a shop, and their shopping habits.
# Start With a Data Set
np.random.seed(2)

x = np.random.normal(3, 1, 100)  # number of minutes before making a purchase.
y = np.random.normal(150, 40, 100) / x  # amount of money spent on the purchase.

# Split Into Train/Test
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

# Display the Training Set
plt.scatter(train_x, train_y)
plt.show()

# Display the Testing Set
plt.scatter(test_x, test_y)
plt.show()

my_model = np.poly1d(np.polyfit(train_x, train_y, 4))
my_line = np.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
# To draw a line through the data points, we use the plot() method of the matplotlib module:
plt.plot(my_line, my_model(my_line), c='r')
plt.show()

# R-squared? R2:
r2 = r2_score(train_y, my_model(train_x))
print(r2)  # The result 0.799 shows that there is a OK relationship.

# Bring in the Testing Set:
r2 = r2_score(test_y, my_model(test_x))
print(r2)  # The result 0.809 shows that the model fits the testing set as well
# Predict Values:

# How much money will a buying customer spend, if she or he stays in the shop for 5 minutes?
print(my_model(5))
print()

np.random.seed(2)

pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmounts = np.random.normal(50.0, 30.0, 100) / pageSpeeds

plt.scatter(pageSpeeds, purchaseAmounts)
plt.xlabel('Page Speeds')
plt.ylabel('Purchase Amounts')
plt.title("Datasets")
plt.show()

trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmounts[:80]
testY = purchaseAmounts[80:]

plt.scatter(trainX, trainY)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title("Train Data")
plt.show()

plt.scatter(testX, testY)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Test Data')
plt.show()

# Polynomial against training data
x = np.array(trainX)
y = np.array(trainY)

p4 = np.poly1d(np.polyfit(x, y, 8))  # using 8th degree polynomial caused overfitting
xp = np.linspace(0, 7, 100)
axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.title('Polynomial against training data')
plt.show()

# Polynomial against testing data
testx = np.array(testX)
testy = np.array(testY)

axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])
plt.scatter(testx, testy)
plt.plot(xp, p4(xp), c='r')
plt.title('Polynomial against testing data')
plt.show()

r2 = r2_score(testY, p4(testX))
print('r2 score on test data:', r2)
r2 = r2_score(np.array(trainY), p4(np.array(trainX)))
print('r2 score on train data:', r2)
