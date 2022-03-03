1. Introduction
   1. 为什么在 intro 中的approximate Q-learning 在很多环境中会失败？
      1. 由于 Q 不断更新，$\operatorname{target}=R\left(s, a, s^{\prime}\right)+\gamma \max Q_{k}\left(s^{\prime}, a^{\prime}\right)$ , 是在不断变化的，同一个(S,A)的target并不一样
      2. 整个 updates 的过程取决于 trajectory，所以不满足梯度下降要求的 iid 分布
   2. High-level idea：make Q-learning look like supervised learning
   3. 2 main ideas to stabilize Q - learning
      1. 用“使用过的experience而不是 online 训练”
         - Experience replay (Lin, 1993$)$.
         - Previously used for better data efficiency.
         - Makes the data distribution more stationary.
      1. 让 target 变化的缓慢一点，因此使用老参数来计算target
        - $L_{i}\left(\theta_{i}\right)=\mathbb{E}_{s, a, s^{\prime}, r \sim D}(\underbrace{r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \theta_{i}^{-}\right)}_{\text {target }}-Q\left(s, a ; \theta_{i}\right))^{2}$


2. Target Network
   1. 重要参数：更新间隔，更新权重。 针对不同问题需要尝试不同参数。
   2. 意图：在梯度下降改变权重时， Q(s,a) 的改变会引起其他 Q(s',a')的改变，假如他们拥有近似的(s,a)。假设 online 更新，那么真的遇到Q(s',a') 时，它由于Q(s,a)的更新，已经比原先值更高了，导致这次可能高估了，也链式反映地影响了其他状态-动作。The network can end up chasing its own tail because of bootstrapping.
   3. 上述的问题，在小网络架构时特别明显。


3. Algorithm
   - deep Q-learning with experience replay. 
   - Initialize replay memory $D$ to capacity $N$
   - Initialize action-value function $Q$ with random weights $\theta$ nitialize target action-value function $\hat{Q}$ with weights $\theta^{-}=\theta$ 
   - For episode $=1, M$ do：
     - Initialize sequence $s_{1}=\left\{x_{1}\right\}$ and preprocessed sequence $\phi_{1}=\phi\left(s_{1}\right)$
     - For $t=1, \mathrm{~T}$ do  
       - With probability $\varepsilon$ select a random action $a_{t}$
       - otherwise select $a_{t}=\operatorname{argmax}_{a} Q\left(\phi\left(s_{t}\right), a ; \theta\right)$
       - Execute action $a_{t}$ in emulator and observe reward $r_{t}$ and image $x_{t+1}$ 
       - Set $s_{t+1}=s_{t}, a_{t}, x_{t+1}$ and preprocess $\phi_{t+1}=\phi\left(s_{t+1}\right)$ 
       - Store transition $\left(\phi_{t}, a_{t}, r_{t}, \phi_{t+1}\right)$ in $D$
       -  Sample random minibatch of transitions $\left(\phi_{j}, a_{j}, r_{j}, \phi_{j+1}\right)$ from $D$
       - Set $y_{j}=\left\{\begin{array}{cc}r_{j} & \text { if episode terminates at step } \mathrm{j}+1 \\ r_{j}+\gamma \max _{a^{\prime}} \hat{Q}\left(\phi_{j+1}, a^{\prime} ; \theta^{-}\right) & \text {otherwise }\end{array}\right.$ 
       - Perform a gradient descent step on $\left(y_{j}-Q\left(\phi_{j}, a_{j} ; \theta\right)\right)^{2}$ with respect to the network parameters $\theta$
       - Every $C$ steps reset $\hat{Q}=Q$


4. 重要实验结论
   1. 使用 Huber Loss，可以避免初期更新幅度过大
      - $L_{\delta}(a)= \begin{cases}\frac{1}{2} a^{2} & \text { for }|a| \leq \delta \\ \delta\left(|a|-\frac{1}{2} \delta\right), & \text { otherwise }\end{cases}$
   2. 不能使用 SGD，而是使用更加复杂的梯度下降法，而且这个操作在 RL 是必须的，必须做很多optimization的优化
   3. 需要逐步降低探索概率，Start $\varepsilon$ at 1 and anneal it to $0.1$ or $0.05$ over the first million frames
   4. stability 在 atari 的验证
$$
\begin{array}{ccccc}
\text { Game } & \begin{array}{c}
\text { With replay, } \\
\text { with target } \mathbf{Q}
\end{array} & \begin{array}{c}
\text { With replay, } \\
\text { without target } \mathbf{Q}
\end{array} & \begin{array}{c}
\text { Without replay, } \\
\text { with target } \mathbf{Q}
\end{array} & \begin{array}{c}
\text { Without replay, } \\
\text { without target } \mathbf{Q}
\end{array} \\
\hline \text { Breakout } & 316.8 & 240.7 & 10.2 & 3.2 \\
\text { Enduro } & 1006.3 & 831.4 & 141.9 & 29.1 \\
\text { River Raid } & 7446.6 & 4102.8 & 2867.7 & 1453.0 \\
\text { Seaquest } & 2894.4 & 822.6 & 1003.0 & 275.8 \\
\text { Space Invaders } & 1088.9 & 826.3 & 373.2 & 302.0
\end{array}
$$

5. 相关优秀模型
   1. Double DQN
      1. 在DQN 中两套参数用于稳定训练，origin 来做动作选择，target 做评估。在更新时的 max(Q)使用的却是target 的选择，显然动作选择应该是让 origin 的
      2. Double DQN 在loss 上做出了改进：$L_{i}\left(\theta_{i}\right)=\mathbb{E}_{s, a, s^{\prime}, r} D\left(r+\gamma Q\left(s^{\prime}, \arg \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \theta\right) ; \theta_{i}^{-}\right)-Q\left(s, a ; \theta_{i}\right)\right)^{2}$
   2. Prioritized Experience Replay
      - Replaying all transitions with equal probability is highly suboptimal.
      - Replay transitions in proportion to absolute Bellman error:$\left|r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \theta^{-}\right)-Q(s, a ; \theta)\right|$
      - 此处可以是根据Bellman差来sample，也可以是对样本做排序，然后再选择
      - 整体的效果是 *显著提速*
   3. Dueling DQN
      1. 这是一个纯粹修改网络结构的技巧：原先我们说，更改某一个动作会影响相似状态动作对，这个网络引入advantage网络，可以更好地分割出每个动作的作用。
      - Value-Advantage decomposition of Q:
        $$
        Q^{\pi}(s, a)=V^{\pi}(s)+A^{\pi}(s, a)
        $$
      - Dueling DQN (Wang et al., 2015):
        $$
        Q(s, a)=V(s)+A(s, a)-\frac{1}{|\mathcal{A}|} \sum_{a=1}^{|\mathcal{A}|} A(s, a)
        $$
   4. Noisy Nets for Exploration
      - Add noise to network parameters for better exploration [Fortunato, Azar, Piot et al. (2017)].
      - Standard linear layer: $\quad y=w x+b$
      - Noisy linear layer: $\quad y \stackrel{\text { def }}{=}\left(\mu^{w}+\sigma^{w} \odot \varepsilon^{w}\right) x+\mu^{b}+\sigma^{b} \odot \varepsilon^{b}$
      - $\varepsilon^{w}$ and $\varepsilon^{\mathrm{b}}$ contain noise.
      - $\sigma^{w}$ and $\sigma^{b}$ are learned parameters that determine the amount of noise.