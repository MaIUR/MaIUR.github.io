# Regularization

### The problem of overfitting

- underfitting : high bias
- overfitting : high variance

If we have too many features, the learned hypothesis may fit the training set very well ($J(\theta)\approx 0$), but fail to generalize to new examples. 

### Addressing overfitting

1. Refuce number of features.
   - Manually select which features to keep
   - Model selection algorithm.
2. Regularization.
   - Keep all the features, but reduce magnitude/values of parameters $\theta_j$.
   - Works well when wehave a lot of features, each of which contributes a bit to predicting $y$.

### Cost function

$$
J(\theta)=\frac{1}{2m}[\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{i=1}^m\theta_j^2]
$$

shrink all the parameters (because we don't know what factors should be shrinked)  

How to choose regularization parameter $\lambda$?

Automatically choose

### Regularized linear regression

##### Gredient decent

Repeat{
$$
\theta_0:=\theta_0-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}\\
\theta_j:=\theta_j-\alpha[\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}+\frac{\lambda}{m}\theta_j]\\
:=\theta_j(1-\alpha\frac{\lambda}{m})-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$
}

$1-\alpha\frac{\lambda}{m}$ is a little bit smaller than 1.

##### Normal equation

$$
X=\begin{bmatrix}(x^{(1)})^T\\\vdots\\(x^{(m)})^T\end{bmatrix}\qquad
y = \begin{bmatrix}y^{(1)}\\\vdots\\x^{(m)}\end{bmatrix}\\
\min_{\theta}J(\theta)\\
\theta=(X^TX+\lambda\underbrace{\begin{bmatrix}0&&&\\&1&&\\&&\ddots&\\&&&1\end{bmatrix}}_{(n+1)\times(n+1)})^{-1}X^Ty
$$

###### Non-invertibility

Suppose $M\le n$, $X^TX$ is non-invertible/singular

Fortunately, if $\lambda>0$, $X^TX+\lambda\begin{bmatrix}0&&&\\&1&&\\&&\ddots&\\&&&1\end{bmatrix}$ is definitly invertable.

### Regularized logistic regression

###### Cost function

$$
J(\theta)=-[\frac{1}{m}\sum_{i=1}^my^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log (1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2
$$

##### Gredient decent

Repeat{
$$
\theta_0:=\theta_0-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}\\
\theta_j:=\theta_j-\alpha[\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}+\frac{\lambda}{m}\theta_j]\\
:=\theta_j(1-\alpha\frac{\lambda}{m})-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$
}

Although it looks quite like the equations in linear regression, it's quite different because $h_\theta(x)$ here is $\frac{1}{1+e^{-\theta^Tx}}$.