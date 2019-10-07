# Name: Griffin Noe 
# Date: 9/20/19
# Course: CSCI 315

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def gradient_descent(x,x_name,y,theta,iterations, learning_rate):
    n = y.size
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    plt.scatter(x,y)
    plt.ylabel("Median Value of House ($000s)")
    plt.xlabel(x_name)
    plt.title("Scatter, No Regression Lines")
    plt.show()
    t=0
    for i in range(iterations):
        prediction = theta[1]*x + theta[0]
        deriv_m = (-2/n) * sum(x * (y-prediction))
        deriv_b = (-2/n) * sum(y - prediction)
        theta[1] = theta[1] - learning_rate * deriv_m
        theta[0] = theta[0] - learning_rate * deriv_b
        MSE = np.square(np.subtract(y,prediction)).mean()
        if(i%(iterations/10)==0):
            plt.title("Regression Run: " + str(t))
            t+=1
            plt.scatter(x,y)
            plt.plot(x, theta[1]*x+theta[0])
            plt.ylabel("Median Value of House ($000s)")
            plt.xlabel(x_name)
            plt.show()
            print("m: " + str(theta[1]) + ", b = " + str(theta[0]))
            print("MSE: " + str(MSE))
    
df = pd.read_csv("housing.csv", header=None, names=["unsplit"])

df = df.unsplit.str.split(expand=True,)

df.columns = ["Crime_Rate", "Lots_Over_25k", "NonRetail_Business", 
                "River", "Nitric_Oxides", "Rooms_Per_Home", 
                "Units_Before_1940", "Distance_To_Employment", 
                "Highway_Accessibility", "Property_Tax", 
                "Student_Teacher_Ratio", "Ethnicity_Demographic", 
                "Lower_Status", "Median_Value"]

y = df["Median_Value"]

gradient_descent(df["Crime_Rate"],"Crime Rate by Town", y, np.random.rand(2),1000,0.01)
gradient_descent(df['Lots_Over_25k'],"Residential land zoned for lots over 25,000 sq.ft.", y, np.random.rand(2),7000,0.001)
gradient_descent(df["NonRetail_Business"],"Proportion of non-retail business acres per town", y, np.random.rand(2),15000,0.001)
gradient_descent(df["River"], "River variable (= 1 if tract bounds river; 0 otherwise)", y, np.random.rand(2),500,0.1)
gradient_descent(df["Nitric_Oxides"], "Nitric oxides concentration (parts per 10 million)",y, np.random.rand(2),5000,0.1)
gradient_descent(df["Rooms_Per_Home"], "Average number of rooms per home",y, np.random.rand(2),30000,0.01)
gradient_descent(df["Units_Before_1940"],"Proportion of owner-occupied units built prior to 1940", y, np.random.rand(2),150000,0.0001)
gradient_descent(df["Distance_To_Employment"], "Weighted distances to five employment centres",y, np.random.rand(2),2000,0.01)
gradient_descent(df["Highway_Accessibility"], "Index of accessibility to radial highways",y, np.random.rand(2),10000,0.001)
gradient_descent(df["Property_Tax"], "Full-value property-tax rate per $10,000",y, np.random.rand(2),100000,0.000001)
gradient_descent(df["Student_Teacher_Ratio"],"Student-teacher ratio by town", y, np.random.rand(2),150000,0.001)
gradient_descent(df["Ethnicity_Demographic"], "Ethncity Demographic", y, np.random.rand(2),50000,0.000001)
gradient_descent(df["Lower_Status"], "Percentage lower status of the population", y, np.random.rand(2),20000,0.001)