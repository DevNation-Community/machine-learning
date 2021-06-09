# Machine Learning
* Definition: A computer program is said to learn from Experience E with respect to some task T and some performance measure P, if its performance on T improves with experience E
* Machine Learning Algorithm:
  * Supervised learning: 
    * We’ll teach the system what to do. 
    * We feed the correct answers to the algorithm and it is supposed to create more of these correct answers.
    * It is also called a Regression problem(which is predicting continuous valued output)
    * Two types: Regression and classification(discrete values) 
  * Unsupervised learning: the system learns on its own
    * A set of data is provided and the ML algo is supposed to find patterns in the data
## Model and Cost Function: Linear Regression with one variable(x)
* Notations: 
  * m = no of training examples 
  * x = “input” variable / features
  * y = “output” variable/”target” variable
  * (x,y) = single training example    

![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1633).png)    

* Training set → learning Algorithm → outputs a function(h - hypothesis)
* We take the training set and feed it to the learning algorithm. The learning algorithm outputs a function - called hypothesis(denoted by h) and this hypothesis is then used to estimate our values.
* The x is then given to the hypothesis and the estimated value is given out by the hypothesis(h)
* h maps from x to y
* How do we represent h? h ϴ (x) = ϴ0 + ϴ1x    
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1634).png)
### Cost Function
* How to make a hypothesis function
  * Choose ϴ0, ϴ1 so that h ϴ (x) is close to y for our training example (x,y)   
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1636).png)
* Cost function(also called squared error function) =     

![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1637).png)
  * In the above formula, the 1/2m means we average out the submission from 1 to m. The h ϴ (x) is the predicted output and y is the actually output. When we subtract the two, we mean the difference between our predicted value and the actual value. The smaller the difference will be the better the hypothesis is.
### Cost Function Intuition I    
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1639).png)
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1640).png)
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1641).png)
### Cost Function Intuition II    
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1642).png)
When we had only one parameter ϴ1 so we got a 2D graph but when we have two parameters ϴ0 and ϴ1 then we will get a 3D graph something like this    
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1643).png)
Instead of using a 3D plot we will be using a contour plot are also called contour figure    
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1644).png)
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1645).png)
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1646).png)
The middle is the min value of J(ϴ0,ϴ1)
### Gradient descent
* Gradient descent for minimizing the cost function J. we'll use gradient descent to minimize other functions as well, not just the cost function J for the linear regression
* Problem setup: 
  * Have some function J(ϴ0,ϴ1) = J(ϴ0,ϴ1….ϴn..)
  * Want min J(ϴ0,ϴ1)
  * Outline:
    * Start with some ϴ0,ϴ1(say ϴ0 = ϴ1 =...... = ϴn = 0)
    * Keep changing ϴ0,ϴ1 to reduce J(ϴ0,ϴ1) until we end up at minimum    
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1651).png)
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1650).png)
* := is used to denote assignment, 𝞪 is the learning rate and what alpha does is that it controls how big a step we take down hill(if 𝞪 is small then we are taking small steps)
* Now, there's one more subtlety about gradient descent which is in gradient descent we're going to update, you know, theta 0 and theta 1, right? So this update takes place for j = 0 and j = 1, so you're gonna update theta 0 and update theta 1. And the subtlety of how you implement gradient descent is for this expression, for this update equation, you want to simultaneously update theta 0 and theta 1. What I mean by that is that in this equation, we're gonna update theta 0 := theta 0 minus something, and update theta 1 := theta 1 minus something. And the way to implement it is you should compute the right hand side, right? Compute that thing for theta 0 and theta 1 and then simultaneously, at the same time, update theta 0 and theta 1.
### Gradient Descent intuition    
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1652).png)
* Derivative at this point does, is basically saying, now let's take the tangent to that point, like that straight line, that red line, is just touching this function, and let's look at the slope of this red line. That's what the derivative is, it's saying what's the slope of the line that is just tangent to the function. The slope of a line is just this height divided by this horizontal thing. Now, this line has a positive slope, so it has a positive derivative. And so my update to theta is going to be theta 1, it gets updated as theta 1, minus alpha times some positive number.
* Alpha learning is always a positive number. And, so we're going to take theta one as it is updated as theta one minus something. So I'm gonna end up moving theta one to the left. I'm gonna decrease theta one, and we can see this is the right thing to do cuz I actually wanna head in this direction. To get me closer to the minimum over there.
* What happens when 𝞪 is to small, then the gradient descent is slow. If 𝞪 is too large, gradient descent can overshoot the min. It may fail to converge, or even diverge.    
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1653).png)
* Gradient descent can converge to a local min, even with the learning rate fixed.    
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1654).png)
  * This is because, when 𝞪 is constant and we are at some point marked pnk in the graph, the derivative(slope) will be something and it will help us make our steps towards the min. Then we come to the next point and the derivative decrease as the slope decrease and hence the steps become smaller, but we do eventually reach to the min point
### Gradient Descent for Linear Regression    
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1655).png)
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1656).png)
A graph for Linear Regression is always going to be a bow shaped function technically called a Convex Function    
![image](https://github.com/schmithvillers/machine-learning/blob/main/Screenshot%20(1644).png)
Batch Gradient Descent: each step of gradient descent uses all the training examples.
