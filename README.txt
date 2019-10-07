Name: Griffin Noe
Date: 9/20/19
Course: CSCI 315
On my honor, I have neither given nor received any unauthorized aid on this project.
Running Instructions: 
    Ensure that housing.csv is in the same folder as the script 
Packages Needed: 
    numpy, pandas, matplotlib
Importing Data: 
    Starting with line 25, I simply read the csv using pandas
    as a data frame. I then split the single column into a column for each 
    variable with names. Because the target variable remains constant, I 
    immediately set y equal to the median value variable. 
Gradient Descent:
    This function takes an x vector, a name for the x vector (only used for plotting)
    a y vector, a randomized theta vector, iterations (aka epochs), and a learning rate. 
    The first thing I do is convert the data points in the x and y vectors to floats as 
    the default dtype is not compatible with the derivative operations. 

    I then jump right into the for loop based on the iterations. Within the 
    loop, I start by making a prediction that is equal to m (theta[1]) times
    the x vector (x) plus b (theta[0]). This is the first, randomized regression
    line. After this, I derive with respect to each variable and store them. 
    This is done because we are doing gradient descent and because we are only
    dealing with two variables, we can simply take the derivative of each variable 
    to find the direction of the greatest slope. 
    Next, I update my m and b values by subtracting the step value times the
    derivative of the existing m or b. This is done because now that we have 
    found the direction of greatest slope, we need to adjust our two variables.
    The derivatives give the biggest increase in the slope so in order to correct
    the variables, we have to subtract their existing values by the derivative of 
    the variable times the learning rate (aka step size). This gives the gradient
    descent. I then calculate the Mean Squared Error. Aptly named as I only had 
    to square the difference between y and the predicted regression line, then find 
    the mean distance from all of the data points, yielding mean square error.

    Next I have a simple if statement that triggers every 1/10 of the way through
    the total number of iterations by checking if the modulo of the current loop
    number by the total number of iterations divided by 10 is zero. Everytime this
    if is triggered: I scatter plot x and y, plot the most recent regression line,
    and display the plot. I then print the value of m, b, and the MSE for that plotted
    regresion line. The final output for every function call is the final m, b, and MSE
    value.

    Finally I simply call gradient descent for every variable with varying iterations
    and learning rates along with randomized thetas. 


Hyperparameter Honing:
    My general philosophy for fine-tuning these two hyperparameters was to make
    the learning rate as large as possible without giving me an error and setting
    iterations to a point where the MSE would more or less stabilize by the tenth 
    print statement. I simply ran each variable and adjusted the learning rate if 
    the program broke or the iterations if the MSE didn't stabilize. For a general 
    note, when presented between fine tuning of the MSE and improving run time
    by giving less iterations, I always favored a fine tuned MSE over a minimized 
    iteration count. This means that some of my iterations are higher than they 
    may need to be but I wanted to make sure that my m, b, and mse were correct.

Sources:
    https://www.kdnuggets.com/2016/11/linear-regression-least-squares-matrix-multiplication-concise-technical-overview.html
    https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/13/lecture-13.pdf
    https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
    https://www.kaggle.com/tentotheminus9/linear-regression-from-scratch-gradient-descent
    https://www.geeksforgeeks.org/python-mean-squared-error/
    https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931