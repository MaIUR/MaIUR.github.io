# SVM

###### Logistic

$$
h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}
$$

denote $z=\theta^Tx$,
$$
h_\theta(x)=g(z)=\frac{1}{1+e^{-z}}
$$
Cost function:
$$
J(\theta)=-(y\log h_\theta(x)+(1-y)\log(1-h_\theta(x)))\\
=-y\log\frac{1}{1+e^{-\theta^Tx}}-(1-y)\log(1-\frac{1}{1+e^{-\theta^Tx}})\\
\min_\theta\frac{1}{m}[\sum^m_{i=1}y^{(i)}(-\log h_\theta(x^{(i)})+(1-y^{(i)})(-\log(1-h_\theta(x^{(i)})))]+\frac{\lambda}{2m}\sum^n_{j=1}\theta^2_j
$$
如果给$\lambda$一个很大的值，就增加了正则项的权重

###### SVM

$$
\min_\theta C[\sum^m_{i=1}y^{(i)}cost_1(\theta^Tx^{(i)})+(1-y^{(i)})cost_0(\theta^Tx^{(i)})]+\frac{1}{2}\sum^n_{j=1}\theta^2_j
$$

这里不用$\lambda$来权衡误差和正则项，而是用$CA+B$

Hypothesis:
$$
h_\theta(x)=\begin{cases} 1& \mbox{if}\ \theta^Tx\ge0\\0&\mbox{otherwise}\end{cases}
$$

#### Large Margin Intuition (大距离分类器)

SVM decision boundary: margin of the support vector machine

only when C is very large.

要使得决策边界和两类别的样本点之间的距离尽可能地大，就要使得$\theta$的模变大，但是在loss function中，正则项又需要最小化$\theta$的值，因此需要选择一个决策边界，使得在增大$p^{(i)}·||\theta||$的绝对值的同时，尽可能第使$\theta$更小。

![image-20200910223716658](F:\study\机器学习\blog\SVM.assets\image-20200910223716658.png)

决策边界是绿色的线，参数向量与决策边界成90°夹角。

- ###### 为什么垂直呢？

  边界线 $\theta_1x_1+\theta_2x_2=0$，斜率为$-\frac{\theta_1}{\theta_2}$

  参数向量为$\theta=\begin{bmatrix}\theta_1 \\ \theta_2\end{bmatrix}$，斜率为$\frac{\theta_2}{\theta_1}$

  相乘为-1因此二者垂直。

要使得$p^{(i)}·||\theta||$的绝对值比较大，对于左边的决策边界，样本点到参数向量的投影（即$p^{(i)}$）比较小，因此$\theta$需要很大，而右边的$p^{(i)}$更大一些，因此可以让$\theta$更小一点

【注意】：简化使得$\theta_0=0$，因此决策边界一定经过原点

#### Kernels

When dealing with a non-linear decision boundary, we would come up with some higher order polynomials such as $x_1X_2,\ x_1^2$ and so on.

The question is, is there a different / better choice of the features?

##### Gaussian Kernal function (similarity)

Given $x$, compute new feature depending on proximity to landmarks $l^{(1)} ,\ l^{(2)} ,\ l^{(3)}$
$$
\mbox{given }x\mbox{, predict "1" when } \theta_0+\theta_1f_1+\theta_2f_2+\theta_3f_3\ge0\\
f_1=similarity(x,l^{(1)})=exp(-\frac{||x-l^{(1)}||^2}{2\sigma^2})=exp(-\frac{\sum_{j=1}^n(x_j-l_j^{(1)})^2}{2\sigma^2})\\
f_2=k(x,l^{(2)})=exp(-\frac{||x-l^{(2)}||^2}{2\sigma^2})
$$
If $x\approx l^{(1)}$: $f_1\approx exp(-\frac{0^2}{2\sigma^2})\approx 1$

If $x$ is far from $l^{(1)}$: $f_1\approx exp(-\frac{(large\ number)^2}{2\sigma^2})\approx 0$

Each landmark $l^{(i)}$ defines a new feature $f_i$, which is a polynomial of $x_i$.

- ###### Where to get the landmarks?

  Given training examples $(x^{(1)},y^{(1)})$, $(x^{(2)},y^{(2)})$, ... , $(x^{(m)},y^{(m)})$, choose $l^{(1)}=x^{(1)}$, $l^{(2)}=x^{(2)}$, ... , $l^{(m)}=x^{(m)}$.

  标记了每个样本到其他样本之间的距离

  For training exmanple $(x^{(i)},y^{(i)})$:
  $$
  x^{(i)}\to f^{(i)}=\begin{bmatrix}f_0=0\\f_1=k(x^{(i)},l^{(1)})\\f_2=k(x^{(i)},l^{(2)})\\ \vdots\\f_m=k(x^{(m)},l^{(m)})\end{bmatrix}
  $$
  上式为特征向量，其中$x_i$那项是1

  因此可以得到$f^{(i)}$， 当$\theta^Tf\ge0$时，预测为"y=1"

- ###### How to get $\theta$

  $$
  \min_\theta C[\sum^m_{i=1}y^{(i)}cost_1(\theta^Tf^{(i)})+(1-y^{(i)})cost_0(\theta^Tf^{(i)})]+\frac{1}{2}\sum^{n=m}_{j=1}\theta^2_j
  $$

  where the regulation term is a little bit different: $-\sum_j\theta_j^2=\theta^t\theta=||\theta||^2=\theta^TM\theta$ (ignoring $\theta_0$)

#### SVM parameters

Large C: lower bias, high varience →overfitting

Small C: Higher bias, low varience →underfitting

Large $\sigma^2$: Features $f_i$ vary more smoothly. 

​                 High bias, low variance.

Small $\sigma^2$: Features $f_i$ vary less smoothly. 

​                 Low bias, high variance.