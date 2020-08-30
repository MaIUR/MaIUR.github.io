# Neural Networks

## 1. Representation

#### Non-linear hypotheses

#### Model representation

- bias unit $x_0=1$

- Sometimes we say this is an artificial neuron with a sigmoid or a logistic activation function. So this activation function in the neuron network terminology is just another term for that function for that non-linearity $g(z)=\frac{1}{1+e^{-z}}$.

- If network has $s_j$ units in layer $j$, $s_{j+1}$ units in layer $j+1$, then $\Theta^{(j)}$ will be of fimension $s_{j+1}\times(s_j+1)$. 

###### Forward propagation: Vectorized inplementation

$$
x=\begin{bmatrix}x_0\\x_1\\\vdots\\x_n\end{bmatrix}\qquad
z^{(j)}=\begin{bmatrix}z_1^{(j)}\\\vdots\\z_n^{(j)}\end{bmatrix}=\Theta^{(j-1)}a^{(j-1)}\qquad
a^{(j)}=g(z^{(j)})
$$

Add $a_0^{j-1}=1$.

## 2. Learning

#### Cost function

###### Logistic regression

$$
J(\theta)=-\frac{1}{m}[\sum_{i=1}^my^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2
$$

###### Neural network

$$
h_\Theta\in\mathbb{R}^K\qquad (h_\Theta(x))_i=i^{th}output\\
J(\Theta)=-\frac{1}{m}[\sum_{i=1}^m\sum_{k=1}^Ky_k^{(i)}\log(h_\Theta(x^{(i)}))_k+(1-y_k^{(i)})\log(1-(h_\Theta(x^{(i)}))_k)]+\frac{\lambda}{2m}\sum_{t=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\Theta_{ji}^{(l)})^2
$$

where $k$ is the number of neurals of the output layer.  

####  Backpropagation algorithm

Given the gradient computation equation above, what we want to do is find the paramater $\Theta$ to try to minimize the $J(\Theta)$. 

###### Need to compute:

- $J(\Theta)$
- $\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)$

![image-20200728231622061](F:\study\机器学习\blog\Neural Networks Representation.assets\image-20200728231622061.png)

###### Backpropagation

Intuition: $\delta_j^{(l)}=$"error" of node $j$ in layer $l$. 

eg. for a 4-layer network
$$
\delta_j^{(4)}=a_j^{(4)}-y_j\\
\delta^{(3)}=(\Theta^{(3)})^T\delta^{(4)}.*g'(z^{(3)})\\
...
$$
where $g'(z^{(3)})$ is the derivative of the activation function $g$ evaluated at the input values given by $z_3$.
$$
g'(z^{(3)})=a^{(3)}.*(1-a^{(3)})
$$
$$
g(z)=\frac{1}{1+e^{-z}}\\
g'(z)=\frac{e^{-z}}{(1+e^{-z})^2}=\frac{e^{-z}}{1+e^{-z}}\times\frac{1}{1+e^{-z}}=(1-a)a
$$



So in the end, 
$$
\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)=a_j^{(l)}\delta_i^{(l+1)}
$$
ignoreing $\lambda_j$ if $\lambda=0$. 

![image-20200728231523178](F:\study\机器学习\blog\Neural Networks Representation.assets\image-20200728231523178.png)

#### Gradient Checking

 ![image-20200827223919837](F:\study\机器学习\blog\Neural Networks.assets\image-20200827223919837.png)

