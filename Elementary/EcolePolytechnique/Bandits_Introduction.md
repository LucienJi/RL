# Bandits

## Introduction

1. Bandits: elements:
	1. **Action** :  $A_{t} \in\{1, \ldots, K\}$, each time we choose the action a, we will receive a reward  $R_{t} \in \mathbb{R}$
	2. **Expeced value for action a** : $q_{*}(a)=\mathbb{E}_{R_{t} \sim \nu_{a}}\left[R_{t} \mid A_{t}=a\right]$
	   1. $\nu\left(R_{t} \mid A_{t}=a\right) \equiv \nu_{a}\left(R_{t}\right)$, here each $\nu_{a}$ corresponds to a **distribution**
	      <img src="/Users/jijingtian/Library/Application Support/typora-user-images/Screen Shot 2022-01-19 at 2.30.45 PM.png" alt="Screen Shot 2022-01-19 at 2.30.45 PM" style="zoom: 33%;" />
	
	3. **Generic bandits framework**: for t =  1 to T:
	      		1. Take <u>action</u> $A_{t} \sim \pi$
	            		2. Observe <u>reward</u> $R_{t}$ (Note: this $R_{t}$ only wrt $A_{t}$ )
	            		3. Update the estimated <u>distribution</u> [if we want to estimate];
	            	  Update the agent <u>policy</u> $\pi$
	            		4. PS: why do need the **sequence of decision ?**
	            	  		1. policy is based on the estimated distribution
	            	  		2. estimated distribution need to be updated according to the interaction [action - reward]
	            	  		3. **Goal of Policy：** help estimate the distribution + maximize the reward 【not the cultimated reward】
	            		4. **Reward and Regret**
	            		1. **mean reward** of an arm (at time t, time t is not important since arm is **stationary**)
	            	   $\mu(a):=\mathbb{E}_{R_{t} \sim \nu_{a}}\left[R_{t} \mid A_{t}=a\right]$
	            		2. **Best reward** : $\mu^{*}:=\max _{a \in \mathcal{A}} \mu(a) \quad a^{*}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} \mu(a)$
	            		3. **Regret: ** for period T and policy $\pi$
	            	  		1. $\mathcal{R}_{T}=\mu^{*} T-\mathbb{E}_{\pi}\left[\sum_{t=1}^{T} R_{t}\right]$
	            	  		2. Best we could have - what we really have
	            	  		3. **Goal: ** minimize regret
	            		5. **Trade-off : Exploitation and Exploration**
	            		1. greedy actions [based on the current knowledge]
	            		2. explore to improve the knowledge
	            		3. Eg:  $\epsilon$-Greedy Algorithm
	            	  		1. play a random arm $a \sim \mathcal{A}$ with probability $\epsilon $ (explore)
	            	  		2. otherwise, greedy action

## Upper Confidence Bounds (UCB) 

1. 理论补充：Tail Probabilities

   1. 针对 某个 arm a，假设 draw T times，we could observe T rewards:
      1. estimated mean reward : $\hat{\mu}=\frac{1}{T} \sum_{t=1}^{T} R_{t}$
      2. [我们会根据 这个 estimated value 进行选择，但是真实 mean 可能<u>大于</u> estimation]
         $P\left(\mu \geq \hat{\mu}+\sqrt{\frac{2 \log (1 / \delta)}{T}}\right) \leq \delta$

2. UCB 

   1. $N_{k}(t)$ be the the number of samples from the $k$-th arm

   2. $\operatorname{UCB}_{k}(t-1, \delta)=\hat{\mu}_{k}(t-1)+\sqrt{\frac{2 \log (1 / \delta)}{N_{k}(t-1)}}$

      1. with confidence level： $\delta$
      2. 2 terms : exploitation + exploration

   3. Generic version:
      <img src="/Users/jijingtian/Library/Application Support/typora-user-images/Screen Shot 2022-01-19 at 3.07.34 PM.png" alt="Screen Shot 2022-01-19 at 3.07.34 PM" style="zoom:33%;" />

      **General idea**: Choosing the arm with the largest UCB implies only playing only arms $k$ **where the true mean $\mu_{k}$ could reasonably be larger than the arms** which have been played often.

      **Good Idea worth spreading :** 是否需要考虑 variance 对于action 选择，而不是只考虑 average reward



## Adverarial bandits setting

1. Adversarial bandits: **no stochastic assumption**, reward is choosen and fixed by an adversarial

2. **Adversarial** : choose sequence of rewards  $\left(x_{t}\right)_{t=1}, \cdots, n$

   1. reward at time t : $x_{t} \in[0,1]^{k}$, **assigned for each arm and fixed**

3. **Learner** : at each round t

   1. choose a distribution of reawrds over arms 
   2. choose the action according to the choosen distribution 
   3. observe the rewards

4. **Policy and Regret**:

   1. **Policy:** given the history, output the distribution over arms

   2. **Regrets: ** $R_{n}(\pi, x)=\max _{i \in[k]} \sum_{t=1}^{n} x_{t i}-\mathbb{E}\left[\sum_{t=1}^{n} x_{t A_{t}}\right]$

      1. 只有 randomization 在 actions

      2. 这是 oblivious adversarial 【不会针对 action 做出改变】

      3. **Our goal： do best in the worst case** ： $R_{n}^{*}(\pi)=\sup _{x \in[0,1]^{n \times k}} R_{n}(\pi, x)$

         1. 什么样的 policy 可以得到一个 sublinear （with n）的 reward

         2. 剧透一下，假如是 deterministic policy，那么 max regret 是 linear 的，所以非常差
            $R_{n}^{*}(\pi) \geq n(1-1 / k)$

         3. **Stochastic regret： ** $R_{n}^{*}(\pi) \geq R_{n}(\pi, \nu)=\underbrace{\max _{i \in[k]} \mathbb{E}\left[\sum_{t=1}^{n}\left(X_{t i}-X_{t A_{t}}\right)\right]}_{\text {stochastic regret }}$
            $\inf _{\pi} R_{n}^{*}(\pi) \geq O(\sqrt{n k})$

            




## Exp3 Algorithm

1. 背景：

   1. 只能看到自己选择的 arm 对应的 reward
   2. 如何估计其他的 arm‘s reward ？
   3. **虽然无法确定 adversarial 的选择，但是还是要通过sample来估计对方的 pattern，假如有 pattern 那么就捕捉然后调整policy，假如真的捕捉不到，那么就纯 random**

2. 提出一个 **arm's reward** 的无偏估计：

   1. $\hat{X}_{t i}=\mathbb{1}_{A_{t}=i} \cdot \frac{X_{t}}{P_{t i}}$
      1. $\mathbb{E}_{t-1} \hat{X}_{t i}=x_{t i}$
      2. $\mathbb{V}_{t-1}\left[\hat{X}_{t i}\right]=x_{t i}^{2} \cdot \frac{1-P_{t i}}{P_{t i}}$
      3. Loss view:
         <img src="/Users/jijingtian/Library/Application Support/typora-user-images/Screen Shot 2022-01-19 at 5.31.24 PM.png" alt="Screen Shot 2022-01-19 at 5.31.24 PM" style="zoom:33%;" />

3. **Exp3** : 一个在线更新的policy + exponential weighting

   1. **importance-weighted estimator** to estimate reward for each arm
   2. 计算 sum：$\hat{S}_{t i}=\sum_{s=1}^{t} \hat{X}_{s i}$
   3. 得到 probability ： 
      $P_{t i}=\frac{\exp \left(\eta \hat{S}_{t-1, i}\right)}{\sum_{j=1}^{k} \exp \left(\eta \hat{S}_{t-1, j}\right)}$
   4. learning rate $\eta$:
      1. 大：close to greedy
      2. 小：close to uniform 

   <img src="/Users/jijingtian/Library/Application Support/typora-user-images/Screen Shot 2022-01-19 at 5.40.19 PM.png" alt="Screen Shot 2022-01-19 at 5.40.19 PM" style="zoom:33%;" />



4. **Regret Bound**： 不同的 learning rate能得到不同的 Regret Bound，也就是说 n round 下不会太差
   1. $\eta=\sqrt{\log (k) /(n k)}$
   2. $R_{n}(\pi, x) \leq 2 \sqrt{n k \log (k)}$
5. Full information case: 就是说：除了自己选择的 arm，也能看到别的 arm 对应的 reward
   1. Same as Exp3, 但是不用 weighted importance samping，而是用真实值 -> Hedge algorithm
   2. Learning rate: $\eta=\sqrt{2 \log (k) / n}$
   3. $R_{n}(\pi, x) \leq \sqrt{2 n \log (k)}$