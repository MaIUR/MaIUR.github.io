# Machine learning system design

##### Recommended approach

- Start with a simple algorithm and plot learning curves to decide if more date, more features, etc. are likely to help.

- Error analysis: manually examine the examples in cross validation set that the algorithm made errors on.

#### Skewed classes (偏斜类)

The number of examples of one class is much much smaller/larger than the other(s).

这种情况下，有的时候误差率降低并不意味着模型更加精确。比如根据症状判断是否癌症的时候（非癌症个数远大于癌症个数），全部预测为非癌症可以使得误差率降低，但着这并不是一个精确的模型。

#### Precision/Recall (查准率/召回率)

<table>
	<tr>
        <th></th>
        <th></th>
	    <th colspan="2">actual class</th>
	</tr >
	<tr >
        <td></td>
        <td></td>
	    <td>0</td>
	    <td>1</td>
	</tr>
	<tr>
        <th rowspan="2">predicted class</th>
	    <td>0</td>
	    <td>true negative</td>
        <td>false negative</td>
	</tr>
	<tr>
	    <td>1</td>
	    <td>false positive</td>
        <td>true positive</td>
	</tr>
</table>
$$
Precision=\frac{\# True\ positives}{\# predicted\ positive}=\frac{\#True\ positive}{\#True\ positive+False\ positive}\\
Recall=\frac{\#True\ positive}{\#Actural\ positive}=\frac{\#True positive}{\#True\ positive+\#False\ negative}
$$

如果增大预测为positive的阈值（即使得预测为positive的条件变得严格，也就是说predicted positive的数量将减少），则会得到higher precision和lower recall

##### How to compare precision/recall numbers?

- Average: $\frac{P+R}{2}$, not very good
- F score: $2\frac{PR}{P+R}$

#### The size of the dataset

Use a learning algorithm with many paramethers (e.g. logistic regression/linear regression with many features; neural network with many hidden units), which can be seen as low bias algorithms.  →$J_{train}(\theta)$ will be small.

Use a very large training set (unlikely to overfit, low variance) →$J_{train}(\theta) \approx J_{test}(\theta)$

Two conditions together → $J_{test}(\theta)$ will be small.