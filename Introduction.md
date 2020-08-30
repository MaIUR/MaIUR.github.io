## Introduction

#### Machine Learning definition

- Arthur Samuel (1959). Machine Learning: Field of study that gives comouters the ability to learn without being explicitly programmed. 在没有明确设置的情况下，使计算机具有学习能力的研究领域。 （Samuel训练了跳棋模型）
- Tom Mitchell (1998). Well-posed Learning Problem: A dcomputer program is said to *learn* from experience E with respect to some task T and some performance measure P, if its perfomace on T, as measured by P, inproves with experience E. 一个适当的学习问题定义如下：计算机程序从经验E中学习，解决某一任务T，进行某一性能度量P，通过P测定在T上的表现因经验E而提高。

#### Machine learning algorithms

- Supervised learning: is teached by human
- Unsupervised learning: learns by itself
- Others: Reinforcement learning, recommender systems

#### Supervised Learning

In supervised learning, we are told what is the "correct answer" that we would have quite liked the algorithms have predicted on that example

- Regression: to predict a continuous valued output
- Classification: to predict a discrete valued outputs

#### Unsupervised Learning

no label or only one label

- clustering algorithm, aka clustering
- cocktail party algorithm

------

### Supplement

1. ##### Semi-supervised learning

   半监督学习在训练阶段结合了大量未标记的数据和少量标签数据。与使用所有标签数据的模型相比，使用训练集的训练模型在训练时可以更为准确，而且训练成本更低。举个例子来说明，我们的朋友Delip Rao在AI咨询公司Joostware工作，他构建了一个使用半监督学习的解决方案，每个类中只需使用30个标签，就可以达到与使用监督学习训练的模型相同的准确度，而在这个监督学习模型中，每个类中需要1360个左右的标签。因此，这个半监督学习方案使得他们的客户能够非常快地将其预测功能从20个类别扩展到110个类别。

   ==为什么使用未标记数据有时可以帮助模型更准确==，关于这一点的体会就是：即使你不知道答案，但你也可以通过学习来知晓，有关可能的值是多少以及特定值出现的频率。

2. ##### SVD (singular value decomposition)

   https://blog.csdn.net/zhongkejingwang/article/details/43053513

   https://www.jianshu.com/p/bcd196497d94

   https://www.cnblogs.com/pinard/p/6251584.html

3. 对称矩阵的性质及证明

   https://wenku.baidu.com/view/ff88dcff102de2bd960588d4.html

