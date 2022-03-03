# Bayesian Networks

## Introduction

1. BN 也是 Probabilistic Graphical Models (PGM) 中的一种
   1. Markov Random Fields 也是 PGM 一种
   2. BN 是有向图， Markov Random Fields  是无向图
   3. 所以 BN 中的 random variable 是可以说 **Parent** 
   4. **是不是说 BN 的图唯一指定了一种 Factorization 方式**：
      $P(A, B, C, D, E)=P(A) P(B) P(C \mid A, B) P(D \mid C) P(E \mid B)$

<img src="/Users/jijingtian/Library/Application Support/typora-user-images/Screen Shot 2022-01-05 at 2.43.36 PM.png" alt="Screen Shot 2022-01-05 at 2.43.36 PM" style="zoom: 50%;" />

2. **Conditional Probability Distribution**:
	 $$
   P\left(X_{j} \mid \operatorname{pa}\left(X_{j}\right)\right)
   $$
   ​										where $\operatorname{pa}\left(X_{j}\right)$ are the parents of $X_{j}$, e.g., $\operatorname{pa}(C)=\{A, B\}$.

3. **Joint Distribution**
    $P(\boldsymbol{X})=\prod_{j=1}^{m} P\left(X_{j} \mid \operatorname{pa}\left(X_{j}\right)\right)$

4. **BN: 一种对于 knowledge 的 representation 方式**：可以在建立的 BN 上做inference

   1. domain expert

   2. data

   3. ###### ==如何根据 data 构建 BN，如何得到 relevant distribution==

5. **Inference 举例**：
   <img src="/Users/jijingtian/Library/Application Support/typora-user-images/Screen Shot 2022-01-05 at 3.01.10 PM.png" alt="Screen Shot 2022-01-05 at 3.01.10 PM" style="zoom:33%;" />   PS：X 是灰色的，说明是 observed
   1. $P(\boldsymbol{Y}=\boldsymbol{y} \mid \boldsymbol{X}=\boldsymbol{x})$
   2. **Query** ： y
   3. **Evidence**： x
   4. eg ： X：观察到咳嗽，Y：接触到病人； Z：被感染 （latent variable）
      <img src="/Users/jijingtian/Library/Application Support/typora-user-images/Screen Shot 2022-01-05 at 3.03.47 PM.png" alt="Screen Shot 2022-01-05 at 3.03.47 PM" style="zoom:50%;" />

		5. **总结：** 在有 query 的前提下，factorisation 成 BN 中的形式



## Dependence and Independence

1. Dependence

   <img src="/Users/jijingtian/Library/Application Support/typora-user-images/Screen Shot 2022-01-05 at 3.24.19 PM.png" alt="Screen Shot 2022-01-05 at 3.24.19 PM" style="zoom:33%;" />

2. Independence:
   <img src="/Users/jijingtian/Library/Application Support/typora-user-images/Screen Shot 2022-01-05 at 3.30.25 PM.png" alt="Screen Shot 2022-01-05 at 3.30.25 PM" style="zoom: 33%;" />



## Application : Multi-Label Classification

<img src="/Users/jijingtian/Library/Application Support/typora-user-images/Screen Shot 2022-01-05 at 3.39.47 PM.png" alt="Screen Shot 2022-01-05 at 3.39.47 PM" style="zoom:33%;" />

1. From statistical point of view: 观察data set， 得到 Statistics，构建BN （可以选择建立 label是 relevant）