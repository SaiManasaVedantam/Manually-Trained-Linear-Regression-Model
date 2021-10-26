# Manually-Trained-Linear-Regression-Model
This project deals with implementing the Linear Regression model on Boston dataset which is available online as well as in Python's Scikit-learn.
I have used Scikit-learn's Boston dataset in the above code.

Python is highly advantageous in using the readily available functions for machine learning algorithms BUT this is in a way leading to a disadvantage for Artificial Intelligence enthusiasts who want to dig deep into details & learn the algorithms with critical optimization techniques.

In the above, I have implemented the Linear Regression model using the following Numerical Optimization Techniques & performed an analysis on them:
1. Gradient Descent
2. Stochastic Gradient Descent (Mini-batch)
3. Stochastic Gradient Descent with Momentum
4. Stochastic Gradient Descent with Nesterov Momentum
5. AdaGrad
6. Adam

Root Mean Square Error (RMSE) is a measure of how spread out these residuals are. In the above code, I used RMSE to make the analysis. If the noise estimated by RMSE is small, it means the model is making more reliable predictions. Otherwise, the model is not highly reliable.

In the code, modifying the values of learning rate (alpha), number of iterations (before convergence), rho values etc. will change the accuracy values. You can play around with those values to make analysis. View the attached sample snapshot (not completely smoothed so that you can try out changing the parameters for better accuracies).

For conceptual understanding, you can watch: https://www.youtube.com/watch?v=nhqo0u1a6fw
