# learning from machine specialization course from deeplearningai.com
import numpy as np
import matplotlib.pyplot as plt
import math

# calculates the squared error cost function
def cost(x, y ,w, b):
    cost = 0
    m = x.shape[0]
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    return cost/(2 * m)

# calculates the derivative of w and b
def gradient(x, y, w, b):
    m = x.shape[0]
    dw = 0
    db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dw += (f_wb - y[i]) * x[i]
        db += f_wb - y[i]
    dw /= m
    db /= m
    return dw, db

# repeats the gradient descent algorithim and records the cost through each iteration
def gradient_descent(x, y ,w_intitial, b_initial, a, iterations):
    w = w_intitial
    b = b_initial
    for i in range(iterations):
        dw, db = gradient(x, y, w, b)
        w = w - a * dw
        b = b - a * db
        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration: {i}\nCost: {cost(x, y, w, b)}\nw: {w:.3e} b: {b:.3e}\n")      
    print(f"Final\nCost: {cost(x, y, w, b)}\nw: {w:.3e} b: {b:.3e}\n")
    return w, b


#data
data = np.genfromtxt('data.csv', delimiter=',', skip_header=1, usecols=(1, 2)) # downloads data from the csv
x_train = data[:, 0] # features
y_train = data[:, 1] # targets
a = 1.0e-2 # learning rate
repetitions = 10000 # iterations
intial_w = 0 # starting w
initial_b = 0 # starting b

# calls the gradient descent function
w, b = gradient_descent(x_train, y_train, intial_w, initial_b, a, repetitions)

# displays the data in the csv and makes a line with w and b
plt.plot(x_train, y_train)
plt.plot(x_train, w * x_train + b)
plt.title("Years of Experience vs. Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
