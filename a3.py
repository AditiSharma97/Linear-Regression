import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def normalEquation (X, y):
	pseudo_inverse = np.linalg.pinv(X)
	w = pseudo_inverse.dot(y)
	return w

def predict (w0, w, x):
	return (w0 + np.dot (x, w))

def grad_desc (X, y, learning_rate):
	m = 7568	
	w0 = 0
	w = np.random.randn (4, 1)
	oldcost = 0
	cost = 100
	while abs (oldcost - cost) > 0.001:
		pred = w0 + np.dot (X, w)
		err = pred - y
		oldcost = cost
		cost = (0.5/m) * (np.sum(np.square(err)))
		dw = np.dot(X.T, err)
		dw.resize ((4,1))
		w0 = w0 - learning_rate * np.sum(err) / m
		w = w - learning_rate * dw / m
	return w0, w

def costCalculation (X_cross_validation, w, y_cross_validation, w0):
	pred = np.zeros (1500)	
	cost = 0
	pred = w0 + np.dot (X_cross_validation, w)
	for i in range (1500):
		diff = y_cross_validation[i] - pred[i]
		diff *= diff
		cost += diff
	cost /= 2 * 1500
	return cost

def l1Regularization (X_training, y_training, learning_rate, X_validation, y_validation):
	cross_validation_losses = []
	m = 6068
	power = 0
	lamb = 1.15
	bestCoeff = 0
	bestW = np.zeros (4)
	bestW0 = 0
	oldcost = 0
	cost = 100
	bestpow = 0
	w_initial = np.random.randn (4,1)
	while power <= 55:
		if power == 0:
			oldcost = cost
		regularization_coeff = lamb ** power
		w0 = 0
		w = w_initial		
		iterations = 3000
		for i in range (iterations):
			pred = w0 + np.dot(X_training, w)
			err = pred - y_training
			cost = (0.5/m) * (np.sum(np.square(err))) + regularization_coeff * np.sum(np.abs(w))
			dw = np.dot (X_training.T, err) + regularization_coeff*(np.sign(w))
			dw.resize ((4,1))
			w0 = w0 - learning_rate * np.sum(err)/m
			w = w - learning_rate * (dw/m)
		newCost = costCalculation (X_validation, w, y_validation, w0)
		cross_validation_losses.append (newCost)
		if (newCost < oldcost):
			bestCoeff = regularization_coeff
			bestW = w
			bestW0 = w0
			bestpow = power
			oldcost = newCost
		power += 1
	return bestCoeff, bestW, bestW0, cross_validation_losses

def l2Regularization (X_training, y_training, learning_rate, X_validation, y_validation):
	cross_validation_losses = []
	m = 6068
	power = 0
	lamb = 1.12
	bestCoeff = 0
	bestW = np.zeros (4)
	bestW0 = 0
	oldcost = 0
	cost = 100
	bestpow = 0
	w_initial = np.random.randn (4,1)
	while power <= 50:
		if power == 0:
			oldcost = cost
		regularization_coeff = lamb ** power
		w0 = 0
		w = w_initial		
		iterations = 3000
		for i in range (iterations):
			pred = w0 + np.dot(X_training, w)
			err = pred - y_training
			cost = (0.5/m) * (np.sum(np.square(err))) + (regularization_coeff/2) * np.sum(w ** 2)
			dw = np.dot (X_training.T, err) + regularization_coeff * w
			dw.resize ((4,1))
			w0 = w0 - learning_rate * np.sum(err)/m
			w = w - learning_rate * (dw/m)
		newCost = costCalculation (X_validation, w, y_validation, w0)
		cross_validation_losses.append (newCost)
		if (newCost < oldcost):
			bestCoeff = regularization_coeff
			bestW = w
			bestW0 = w0
			bestpow = power
			oldcost = newCost
		power += 1
	return bestCoeff, bestW, bestW0, cross_validation_losses

def normalise (X_training, X_testing):
	mu = np.mean (X_training, axis = 1, keepdims = 1)
	#sigma2 = np.var (X_training, axis = 1, keepdims = 1)
	sigma = np.std(X_training, axis = 1, keepdims = 1)	
	X_training = X_training - mu
	X_training /= sigma
	X_testing = X_testing - mu
	X_testing /= sigma
	return X_training, X_testing

design_parameters = 4
#testingDataSize = 2000
#training_data_size = 7568
excel_file = 'Folds5x2_pp.xlsx'
data = pd.read_excel(excel_file)
ones = np.ones(7568)
ones = ones[:, None]
oldphi = data.iloc[0:7568, 0:design_parameters].values #input raw data values (without concatenating 1s column)
phi = np.hstack ((ones,oldphi)) #input data values after concatenating 1s column
t = data.iloc[0:7568, design_parameters].values #y values

W = normalEquation (phi, t)
ones = np.ones(2000)
ones = ones[:, None]
oldtesting_phi = data.iloc[7568:9570, 0:design_parameters].values #testing data values (without concatenating 1s column)
testing_phi = np.hstack ((ones,oldtesting_phi)) #testing data values after concatenating 1s column
testing_t = data.iloc[7568:9570, design_parameters].values #testing data y values
testing_y = np.zeros (2000) #predicted y values after normal equation method
rmse = 0
for i in range (2000):
	for j in range (4):
		testing_y[i] += (W[j+1] * testing_phi[i][j+1])
	testing_y[i] += W[0]
	diff = testing_y[i] - testing_t[i]
	diff *= diff
	rmse += diff
rmse /= 2000
print ("Rmse for normal equations")
print (rmse)

phi_t = oldphi.T
testing_phi_t = oldtesting_phi.T
phi_t, testing_phi_t = normalise (phi_t, testing_phi_t) #normalising input data (training and testing)
phi = phi_t.T #normalised training data (without concatenating 1s column)
testing_phi = testing_phi_t.T #normalised testing data (without concatenating 1s column)
ones = np.ones(7568)
ones = ones[:, None]	
phi = np.hstack ((ones, phi)) #normalised training data after concatenating 1s column

w0, w = grad_desc (phi_t.T, t[:, None], 0.5)
pred = predict (w0, w, testing_phi)
rms = 0
for i in range (2000):
	diff = pred[i] - testing_t[i]
	diff *= diff
	rms += diff
rms /= 2000
print ("gradient desc rmse")
print (rms)

trainingDataForRegularization = phi_t.T[:6068]
y_training = t[:6068, None]
learning_rate = 0.5
validationDataForRegularization = phi_t.T[6068:7568]
y_validation = t[6068:7568]

l1_cv_losses = []
lamb, l1_w, l1_w0, l1_cv_losses = l1Regularization (trainingDataForRegularization, y_training, learning_rate, validationDataForRegularization, y_validation)
phi_t = phi.T
phi_t = phi.T [1:]
phi_t = phi_t.T
print ("l1 lambda")
print (lamb)
testing_y = predict (l1_w0, l1_w, testing_phi)
rmse = 0
for i in range (2000):
	diff = testing_y[i] - testing_t[i]
	diff *= diff
	rmse += diff
rmse /= 2000

print ("l1 rmse")
print (rmse)

l2_cv_losses = []
lamb, l2_w, l2_w0, l2_cv_losses = l2Regularization (trainingDataForRegularization, y_training, learning_rate, validationDataForRegularization, y_validation)
print ("l2 lambda")
print (lamb)
testing_y = predict (l2_w0, l2_w, testing_phi)
rmse = 0
for i in range (2000):
	diff = testing_y[i] - testing_t[i]
	diff *= diff
	rmse += diff
rmse /= 2000

print ("l2 rmse")
print (rmse)

l1_lambds = np.logspace(0, 55, num = 56, base=1.15)
fig1, ax1 = plt.subplots()
ax1.plot(l1_lambds, l1_cv_losses)
ax1.set(xlabel='Lambda', ylabel='Cross-Validation Loss', title='L1 Regularization')
ax1.grid()
plt.xscale('log')
plt.show()
    
l2_lambds = np.logspace (0, 50, num = 51, base = 1.12)
fig2, ax2 = plt.subplots()
ax2.plot(l2_lambds, l2_cv_losses)
ax2.set(xlabel='Lambda', ylabel='Cross-Validation Loss', title='L2 Regularization')
ax2.grid()
plt.xscale('log')
plt.show()
