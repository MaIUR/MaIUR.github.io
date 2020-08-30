# Logistic Regression

### Classification

- Binary classification problem: $y\in\{0,\, 1\}$, 0 - Negative Class, 1 - Positive Class
- Multiple classification problem

$h_\theta(x)$ can be $>1$ or $<0$, while logistic regression: $0\le h_\theta(x)\le 1$ (although there is a 'regression' in the name of logistic regression, it is actually a classfication algorithm)

### Hypothesis Representation

- sigmoid function (logistic function): $g(x)=\frac{1}{1+e^{-z}}$ , where $z=\theta^\mathrm{T}x$ 

  - what to do is fit the parameters to the data

- Interpretation of Hypothesis Output:

  $h_\theta(x)=P(y=1|x;\theta)$  probability that y=1, given x, parameterized by $\theta$.   

  $P(y=1|x;\theta)+P(y=1|xl\theta)=1$

### Decision Boundary

The hypothesis function is going to predict $y=1$ whenever $\theta^\mathrm{T}x\ge0$.

$\theta$ defines the decision boundary.

### Cost function

How to choose parameter $\theta$?

- Linear regression: $J(\theta)=\frac{1}{m}\sum_{i=1}^{m}\frac{1}{2}(h_\theta(x^{(i)})-y^{(i)})^2$
- Define cost function: $Cost(h_\theta(x),y)=\frac{1}{2}(h_\theta(x)-y)^2$, which is the learning algorithm will have to pay

However, this cost function is for linear regression. If we use this particular cost function, this would be a non-convex function of the parameters $\theta$.
$$
\mathrm{for}\;y=1\qquad Cost(h_\theta(x),y)=\begin{cases}-\log(1-h_\theta(x))& \mathrm{if}\;y=0\\
-\log(h_\theta(x))& \mathrm{if}\;y=1
\end{cases}\\
\mathrm{for}\;y=0\qquad Cost(h_\theta(x),y)=\begin{cases}-\log(h_\theta(x))& \mathrm{if}\;y=0\\
-\log(1-h_\theta(x))& \mathrm{if}\;y=1
\end{cases}
$$
This cost function captures intution that if $h_\theta(x)=0$, but $y=1$, we'll penalize learning algorithm by a very large cost.
$$
\mathrm{for}\;y=1\qquad Cost(h_\theta(x),y)=-y\log(h_\theta(x))-(1-y)\log(1-h_\theta(x))\\
\mathrm{for}\;y=0\qquad Cost(1-h_\theta(x),y)=-y\log(h_\theta(x))-(1-y)\log(h_\theta(x))\\
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}Cost(h_\theta(x^{(i)}),y^{(i)})
$$
The equation can be derived from statistics using the principle of maximum likelihood estimation.

> MLE?

What we need to do is find the $\theta$ that minimize the $J(\theta)$.

One way to do this is using gradient decent.

[Note: feature scal can also be applied to logistic regression.]

### Advanced optimization

- Gradient decent
- Conjugate gradient
- BFGS 共轭梯度法
- L-BFGS

##### Advantages:

- No need to manually pick $\alpha$. (Clever inner-loop called a line search algorithm that automatically tries out different values for the learning rate $\alpha$ and automatically picks a good one, even different values for different iterations.)
- Often faster than gradient decent

##### Disadvanteges:

- More complex

### Multi-class classification One-vs-all

Separete the problem into n single classfication problems.

