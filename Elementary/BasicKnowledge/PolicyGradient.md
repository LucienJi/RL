# Baisc Idea
1. Introduction
   1. Idea 是policy optimization 而不是和 Q，V iteration 一样的 dynamic programming
      1. Policy Optimization：
         1. 直接优化策略
         2. 更加适合循环网络等复杂结构
         3. 可以加上 auxiliary task
      2. Dynamic Programming
         1. 并不是直接得出策略，而是探索状态和动作
         2. 优点是适合：exploration，off-policy，sample-efficiency
   2. 为什么要使用 stochastic policy？
      1. V : 没有动作信息
      2. Q ：argmax 在 continuous 环境中不适用


2. 数学推导
   1. 最大化目标：$U(\theta)=\mathrm{E}\left[\sum_{t=0}^{H} R\left(s_{t}, u_{t}\right) ; \pi_{\theta}\right]=\sum_{\tau} P(\tau ; \theta) R(\tau)$
      1. $\tau$ denote a state-action sequence $s_{0}, u_{0}, \ldots, s_{H}, u_{H}$，所以最终目标不是最大化最后一个reward，而是整个 trajectory 的非递减累加和
      2. 给定一个策略网络，由于随机性会生成不同的 trajectory，目的是以已诊断trajector中搜到的reward为考虑对象，最大化所有的trajectory reward
      3. 我们可以sample 很多 trajectory 来estimate 这个objective function， 我们最终想要的是
   2. 梯度 
	$$
	\begin{aligned} \nabla_{\theta} U(\theta) &=\nabla_{\theta} \sum_{\tau} P(\tau ; \theta) R(\tau) \\ &=\sum_{\tau} \nabla_{\theta} P(\tau ; \theta) R(\tau) \\ &=\sum_{\tau} \frac{P(\tau ; \theta)}{P(\tau ; \theta)} \nabla_{\theta} P(\tau ; \theta) R(\tau) \\ &=\sum_{\tau} P(\tau ; \theta) \frac{\nabla_{\theta} P(\tau ; \theta)}{P(\tau ; \theta)} R(\tau) \\ &=\sum_{\tau} P(\tau ; \theta) \nabla_{\theta} \log P(\tau ; \theta) R(\tau) \end{aligned}
	$$

      - 上述操作的优点是，重新转写出期望，从而可以使用sample-based来 approximate 这个梯度  
	$$
   \nabla_{\theta} U(\theta) \approx \hat{g}=\frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right) R\left(\tau^{(i)}\right)
	$$
   
      - 可以理解为，这个梯度会根据 reward 的值来改变 “一整个path” 出现的概率
      - 将trajectory概率转换到 state 和 action 上：
	$$
   \begin{aligned} \nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right) &=\nabla_{\theta} \log [\prod_{t=0}^{H} \underbrace{P\left(s_{t+1}^{(i)} \mid s_{t}^{(i)}, u_{t}^{(i)}\right)}_{\text {dynamics model }} \cdot \underbrace{\pi_{\theta}\left(u_{t}^{(i)} \mid s_{t}^{(i)}\right)}_{\text {policy }}] \\ &=\nabla_{\theta}\left[\sum_{t=0}^{H} \log P\left(s_{t+1}^{(i)} \mid s_{t}^{(i)}, u_{t}^{(i)}\right)+\sum_{t=0}^{H} \log \pi_{\theta}\left(u_{t}^{(i)} \mid s_{t}^{(i)}\right)\right] \\ &=\nabla_{\theta} \sum_{t=0}^{H} \log \pi_{\theta}\left(u_{t}^{(i)} \mid s_{t}^{(i)}\right) \\ &=\sum_{t=0}^{H} \underbrace{\nabla_{\theta} \log \pi_{\theta}\left(u_{t}^{(i)} \mid s_{t}^{(i)}\right)}_{\text {no dynamics model required!! }} \end{aligned}
	$$
      - 优点是：梯度和模型完全不管，唯一需要的就是 policy
    		- **总结一下**：我们可以通过 sampling 的方式获得“改变path出现概率”的梯度的*无偏估计*

        $$
        \hat{g}=\frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right) R\left(\tau^{(i)}\right)
        $$
        Here:
        $$
        \nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right)=\sum_{t=0}^{H} \underbrace{\nabla_{\theta} \log \pi_{\theta}\left(u_{t}^{(i)} \mid s_{t}^{(i)}\right)}_{\text {no dynamics model required!! }}
        $$
   
    1. Importance Sampling:
       1. 考虑到：采样时和更新的所面对的是（可能是）不同参数
       2. $\begin{aligned} U(\theta) &=\mathbb{E}_{\tau \sim \theta_{\text {old }}}\left[\frac{P(\tau \mid \theta)}{P\left(\tau \mid \theta_{\text {old }}\right)} R(\tau)\right] \\ \nabla_{\theta} U(\theta) &=\mathbb{E}_{\tau \sim \theta_{\text {old }}}\left[\frac{\nabla_{\theta} P(\tau \mid \theta)}{P\left(\tau \mid \theta_{\text {old }}\right)} R(\tau)\right] \\\left.\nabla_{\theta} U(\theta)\right|_{\theta=\theta_{\text {old }}} &=\mathbb{E}_{\tau \sim \theta_{\text {old }}}\left[\frac{\left.\nabla_{\theta} P(\tau \mid \theta)\right|_{\theta_{\text {old }}}}{P\left(\tau \mid \theta_{\text {old }}\right)} R(\tau)\right] \\ &=\mathbb{E}_{\tau \sim \theta_{\text {old }}}\left[\left.\nabla_{\theta} \log P(\tau \mid \theta)\right|_{\theta_{\text {old }}} R(\tau)\right] \end{aligned}$

***
1. 上述内容，构成了最基础的 REINFORCE算法： 
   1. sample 完整 trajectory
   2. 累加平均多段trajectory，累加所有reward。 $\nabla_{\theta} J(\theta) \approx \sum_{i}\left(\sum_{t} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{t}^{i} \mid \mathbf{s}_{t}^{i}\right)\right)\left(c\right)$
   3. $\theta \leftarrow \theta+\alpha \nabla_{\theta} J(\theta)$ 


2. 增加有关 Continuous Action 如何计算策略梯度：
   1. 输出的是 mean and std，最终概率是 $=\mathcal{N}\left(f_{\text {neural network }}\left(\mathbf{s}_{t}\right) ; \Sigma\right)$
   2. $\log \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)=-\frac{1}{2}\left\|f\left(\mathbf{s}_{t}\right)-\mathbf{a}_{t}\right\|_{\Sigma}^{2}+\mathrm{const}$
   3. $\nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)=-\frac{1}{2} \Sigma^{-1}\left(f\left(\mathbf{s}_{t}\right)-\mathbf{a}_{t}\right) \frac{d f}{d \theta}$


3. 核心问题在哪里？所有的 PG 算法的最大问题在这里就体现了：**High Variance**
   1. 什么是 high variance？新的sample参与更新后会对原先的trajectory probability产生巨大的影响，甚至说，假如有两个trajectory 的reward就是 0，那么梯度根本不会改变 
***

1. Improvement for policy gradient
   1. Baseline： $\nabla U(\theta) \approx \hat{g}=\frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right)\left(R\left(\tau^{(i)}\right)-b\right)$
      1. baseline doesn’t depend on action in logprob(action)
      2. 如何选择Baseline
         - Constant baseline: $b=\mathbb{E}[R(\tau)] \approx \frac{1}{m} \sum_{i=1}^{m} R\left(\tau^{(i)}\right)$
         - Optimal Constant baseline: $\quad b=\frac{\sum_{i}\left(\nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right)\right)^{2} R\left(\tau^{(i)}\right)}{\sum_{i}\left(\nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right)\right)^{2}}$
         - Time-dependent baseline: $\quad b_{t}=\frac{1}{m} \sum_{i=1}^{m} \sum_{k=t}^{H-1} R\left(s_{k}^{(i)}, u_{k}^{(i)}\right)$
         - State-dependent expected return:
            $$
            b\left(s_{t}\right)=\mathbb{E}\left[r_{t}+r_{t+1}+r_{t+2}+\ldots+r_{H-1}\right]=V^{\pi}\left(s_{t}\right)
            $$
   2. Temporal Structure - Causality: policy at t' cannot affect reward at time t when t < t'
      1. Removing terms that don’t depend on current action can **lower variance**:
      2. 将trajectory 拆开
         - $\begin{aligned} \hat{g} &=\frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right)\left(R\left(\tau^{(i)}\right)-b\right) \\ &=\frac{1}{m} \sum_{i=1}^{m}\left(\sum_{t=0}^{H-1} \nabla_{\theta} \log \pi_{\theta}\left(u_{t}^{(i)} \mid s_{t}^{(i)}\right)\right)\left(\sum_{t=0}^{H-1} R\left(s_{t}^{(i)}, u_{t}^{(i)}\right)-b\right) \\ &=\frac{1}{m} \sum_{i=1}^{m}\left(\sum_{t=0}^{H-1} \nabla_{\theta} \log \pi_{\theta}\left(u_{t}^{(i)} \mid s_{t}^{(i)}\right)\left[\left(\sum_{k=0}^{t-1} R\left(s_{k}^{(i)}, u_{k}^{(i)}\right)\right)+\left(\sum_{k=t}^{H-1} R\left(s_{k}^{(i)}, u_{k}^{(i)}\right)\right)-b\right]\right) \end{aligned}$
         - $\left.\sum_{k=0}^{t-1} R\left(s_{k}^{(i)}, u_{k}^{(i)}\right)\right)$ ，不依赖于 $u_{t}^{(i)}$,依赖的那一部分可看作是 Q 函数的一次 estimate
         - b: 可以依赖于 $s_{t}^{(i)}$
         - 总结一下：$\frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{H-1} \nabla_{\theta} \log \pi_{\theta}\left(u_{t}^{(i)} \mid s_{t}^{(i)}\right)\left(\sum_{k=t}^{H-1} R\left(s_{k}^{(i)}, u_{k}^{(i)}\right)-b\left(s_{t}^{(i)}\right)\right)$

    1. 为了 Baseline 估计 Value function
       1. Mont-Carlo 法：not sample efficiency，but not baised
            - Collect trajectories $\tau_{1}, \ldots, \tau_{m}$
            - Regress against empirical return:
                $$
                \phi_{i+1} \leftarrow \underset{\phi}{\arg \min } \frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{H-1}\left(V_{\phi}^{V \pi}\left(s_{t}^{(i)}\right)-\left(\sum_{k=t}^{H-1} R\left(s_{k}^{(i)}, u_{k}^{(i)}\right)\right)\right)^{2}
                $$
       2. TD equation：sample efficiency but baised
            1. Bellman Equation for $V^{\pi}$
            $$
            V^{\pi}(s)=\sum_{u} \pi(u \mid s) \sum_{s^{\prime}} P\left(s^{\prime} \mid s, u\right)\left[R\left(s, u, s^{\prime}\right)+\gamma V^{\pi}\left(s^{\prime}\right)\right]
            $$
           1. 自助更新
         - Collect data $\left\{s, u, s^{\prime}, r\right\}$
         - Fitted V iteration:
            $$
            \phi_{i+1} \leftarrow \min _{\phi} \sum_{\left(s, u, s^{\prime}, r\right)}\left\|r+V_{\phi_{i}}^{\pi}\left(s^{\prime}\right)-V_{\phi}(s)\right\|_{2}^{2}+\lambda\left\|\phi-\phi_{i}\right\|_{2}^{2}
            $$

 1. Vanilla Policy Gradient
   - Initialize policy parameter $\theta$, baseline $b$
   - for iteration $=1,2, \ldots$ do
     - Collect a set of trajectories by executing the current policy 
     - At each timestep in each trajectory, compute the return $R_{t}=\sum_{t^{\prime}=t}^{T-1} \gamma^{t^{\prime}-t} r_{t^{\prime}}$, and the advantage estimate $\hat{A}_{t}=R_{t}-b\left(s_{t}\right)$ 
     - Re-fit the baseline, by minimizing $\left\|b\left(s_{t}\right)-R_{t}\right\|^{2}$, summed over all trajectories and timesteps.
     - Update the policy, using a policy gradient estimate $\hat{g}$, which is a sum of terms $\nabla_{\theta} \log \pi\left(a_{t} \mid s_{t}, \theta\right) \hat{A}_{t}$ end for


# A3C and GAE

1. Variance Reduction（似乎更加向往拥有稳定的估计）
   1. Gradient：$\frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{H-1} \nabla_{\theta} \log \pi_{\theta}\left(u_{t}^{(i)} \mid s_{t}^{(i)}\right)\left(\sum_{k=t}^{H-1} R\left(s_{k}^{(i)}, u_{k}^{(i)}\right)-V^{\pi}\left(s_{k}^{(i)}\right)\right)$
   2. $\sum_{k=t}^{H-1} R\left(s_{k}^{(i)}, u_{k}^{(i)}\right)$ is estimate of Q from a *single roll out*.
      1. 为什么这个 sampling 不好？这的确 low bias，因为绝对是真实值，但是high variance，缺少了 generalization 的能力
      2. 可以：使用 discount ； 使用function approximation 来降低 variance

2. Multi-step 
      - $$
        \begin{aligned}
        Q^{\pi, \gamma}(s, u) &=\mathbb{E}\left[r_{0}+\gamma r_{1}+\gamma^{2} r_{2}+\cdots \mid s_{0}=s, u_{0}=u\right] \\
        &=\mathbb{E}\left[r_{0}+\gamma V^{\pi}\left(s_{1}\right) \mid s_{0}=s, u_{0}=u\right] \\
        &=\mathbb{E}\left[r_{0}+\gamma r_{1}+\gamma^{2} V^{\pi}\left(s_{2}\right) \mid s_{0}=s, u_{0}=u\right] \\
        &=\mathbb{E}\left[r_{0}+\gamma r_{1}++\gamma^{2} r_{2}+\gamma^{3} V^{\pi}\left(s_{3}\right) \mid s_{0}=s, u_{0}=u\right] \\
        &=\cdots
        \end{aligned}
        $$
      - Async Advantage Actor Critic (A3C) [Mnih et al, 2016]
        $\hat{Q}$ one of the above choices (e.g. $\mathrm{k}=5$ step lookahead)

3. GAE
   1. 小孩子才做选择，成年人全都要，用指数加权的方式，将 look ahead k = 1 ：n 全都加起来
   2. Recap
      1. Return Function：$R(\tau)=r_{0}+\gamma r_{1}+\ldots=\sum_{t=0}^{\infty} \gamma^{t} r_{t}$
      2. Advantage：$A(s, a)=E_{\tau}\left[R(\tau) \mid s_{0}=s, a_{0}=a\right]-V(s)$ = $Q^{\pi}\left(s_{t}, a_{t}\right)-V^{\pi}\left(s_{t}\right)$
      3. TD residual(这不是用来更新 V 的方式，更新V用的是 max，此处理解为 Q - V)：$\delta_{t}^{V}=r_{t}+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right)$

    3. Rewrite the Advantage:
       1. $\hat{A}_{t}^{(k)}=\sum_{l=0}^{k-1} \gamma^{l} \delta_{t+l}^{V}$
       2. Weighted: GAE: $\hat{A}_{t}^{G A E(\gamma, \lambda)}=\sum_{l=0}^{\infty}(\gamma \lambda)^{l} \delta_{t+l}^{V}$


4. Algorithm
   - Init $\pi_{\theta_{0}} V_{\phi_{0}}^{\pi}$
   - Collect roll-outs $\left\{\mathrm{s}, \mathrm{u}, \mathrm{s}^{\prime}, \mathrm{r}\right\}$ and $\hat{Q}_{i}(s, u)$
   $$
    \begin{aligned}
    \text { Update: } \quad & \phi_{i+1} 
    & \leftarrow \min _{\phi} \sum_{\left(s, u, s^{\prime}, r\right)}\left\|\hat{Q}_{i}(s, u)-V_{\phi}^{\pi}(s)\right\|_{2}^{2}+\kappa\left\|\phi-\phi_{i}\right\|_{2}^{2} \\
    & \theta_{i+1} & \leftarrow \theta_{i}+\alpha \frac{1}{m} \sum_{k=1}^{m} \sum_{t=0}^{H-1} \nabla_{\theta} \log \pi_{\theta_{i}}\left(u_{t}^{(k)} \mid s_{t}^{(k)}\right)\left(\hat{Q}_{i}\left(s_{t}^{(k)}, u_{t}^{(k)}\right)-V_{\phi_{i}}^{\pi}\left(s_{t}^{(k)}\right)\right)
    \end{aligned}
   $$

    - Variation：
      - $\phi_{i+1} \leftarrow \min _{\phi} \sum_{\left(s, u, s^{\prime}, r\right)}\left\|r+V_{\phi_{i}}^{\pi}\left(s^{\prime}\right)-V_{\phi}(s)\right\|_{2}^{2}+\lambda\left\|\phi-\phi_{i}\right\|_{2}^{2}$
      - $\theta_{i+1} \leftarrow \theta_{i}+\alpha \frac{1}{m} \sum_{k=1}^{m} \sum_{t=0}^{H-1} \nabla_{\theta} \log \pi_{\theta_{i}}\left(u_{t}^{(k)} \mid s_{t}^{(k)}\right)\left(\sum_{t^{\prime}=t}^{H-1} r_{t^{\prime}}^{(k)}-V_{\phi_{i}}^{\pi}\left(s_{t^{\prime}}^{(k)}\right)\right)$


# TRPO
*****
1. 首先在描述 TRPO 和 PPO 之前，再次复述一下基础policy Gradient 的缺点（REINFORCEMENT 的缺点 ）
   - 这是 on - policy 的算法，因此必须sample 一次trajectory ，然后Gradient一次。假如连续gradient，那么第二次gradient的时候，这个trajectory的数据并不是这个参数能够得到的。

2. Recap： Importance Sampling
   - $\begin{aligned} E_{x \sim p(x)}[f(x)] &=\int p(x) f(x) d x \\ &=\int \frac{q(x)}{q(x)} p(x) f(x) d x \\ &=\int q(x) \frac{p(x)}{q(x)} f(x) d x \\ &=E_{x \sim q(x)}\left[\frac{p(x)}{q(x)} f(x)\right] \end{aligned}$
   - $J(\theta)=E_{\tau \sim \bar{p}(\tau)}\left[\frac{p_{\theta}(\tau)}{\bar{p}(\tau)} r(\tau)\right]$

3. 尝试将 off - policy 改入 REINFORCEMENT 中看看会发生什么？
   1. $\theta'$ 是当前参数，而$\theta$ 是旧的、用于sample 的参数 ： $\frac{p_{\theta^{\prime}}(\tau)}{p_{\theta}(\tau)}=\frac{\prod_{t=1}^{T} \pi_{\theta^{\prime}}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)}{\prod_{t=1}^{T} \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)}$
   2. $\nabla_{\theta^{\prime}} J\left(\theta^{\prime}\right)=E_{\tau \sim p_{\theta}(\tau)}\left[\frac{p_{\theta^{\prime}}(\tau)}{p_{\theta}(\tau)} \nabla_{\theta^{\prime}} \log \pi_{\theta^{\prime}}(\tau) r(\tau)\right] \quad$ when $\theta \neq \theta^{\prime}$
   3. 上述公式的化简不难，但是 causality 会很难写 $=E_{\tau \sim p_{\theta}(\tau)}\left[\left(\prod_{t=1}^{T} \frac{\pi_{\theta^{\prime}}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)}{\pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)}\right)\left(\sum_{t=1}^{T} \nabla_{\theta^{\prime}} \log \pi_{\theta^{\prime}}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\right)\left(\sum_{t=1}^{T} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right)\right]$
   4. 考虑Causality：$E_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=1}^{T} \nabla_{\theta^{\prime}} \log \pi_{\theta^{\prime}}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\left(\prod_{t^{\prime}=1}^{t} \frac{\pi_{\theta^{\prime}}\left(\mathbf{a}_{t^{\prime}} \mid \mathbf{s}_{t^{\prime}}\right)}{\pi_{\theta}\left(\mathbf{a}_{t^{\prime}} \mid \mathbf{s}_{t^{\prime}}\right)}\right)\left(\sum_{t^{\prime}=t}^{T} r\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right)\left(\prod_{t^{\prime \prime}=t}^{t^{\prime}} \frac{\pi_{\theta^{\prime}}\left(\mathbf{a}_{t^{\prime \prime}} \mid \mathbf{s}_{t^{\prime \prime}}\right)}{\pi_{\theta}\left(\mathbf{a}_{t^{\prime \prime}} \mid \mathbf{s}_{t^{\prime \prime}}\right)}\right)\right)\right]$
   5. $\prod_{t^{\prime}=1}^{t} \frac{\pi_{\theta^{\prime}}\left(\mathbf{a}_{t^{\prime}} \mid \mathbf{s}_{t^{\prime}}\right)}{\pi_{\theta}\left(\mathbf{a}_{t^{\prime}} \mid \mathbf{s}_{t^{\prime}}\right)}$， 首先这个单领出来说明“未来的动作概率不会影响到这一步的更新”。但问题来了，这个累乘作用在梯度上，是会消失或者爆炸的。
   6. $\prod_{t^{\prime \prime}=t}^{t^{\prime}} \frac{\pi_{\theta^{\prime}}\left(\mathbf{a}_{t^{\prime \prime}} \mid \mathbf{s}_{t^{\prime \prime}}\right)}{\pi_{\theta}\left(\mathbf{a}_{t^{\prime \prime}} \mid \mathbf{s}_{t^{\prime \prime}}\right)}$，这个呢理论上是要累乘后给到reward to go 上的，，但是实际应用时可以讨论一下


   7. 迷之近似：$\prod_{t^{\prime}=1}^{t} \frac{\pi_{\theta^{\prime}}\left(\mathbf{a}_{t^{\prime}} \mid \mathbf{s}_{t^{\prime}}\right)}{\pi_{\theta}\left(\mathbf{a}_{t^{\prime}} \mid \mathbf{s}_{t^{\prime}}\right)}$ ~ $\frac{\pi_{\theta^{\prime}}\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)}{\pi_{\theta}\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)}$， 用 marginal 代替了累乘概率
*****

1. Limitation of Vanilla PG
   1. 如何选择 stepsize?
      - Policy 直接决定了 observation 和 reward 的 distribution，所以更新policy会导致不能 iid
      - Step 会影响 **visiting distribution**，step too far -> bad policy -> bad collection -> never recover
   2. Sampling Efficiency:
      - Only one gradient step per environment sample

2. Limitation of RL
   1. Supervised Learning 本质上是最优化问题：最小化误差
   2. RL：
      - Q - learning:最接近最优化问题，因为目的是拟合 Q 函数，完全可以off policy 去训练，唯一的要求就是收集所有的transition；缺点是RL的目的不是Q函数，而是策略的performance
      - PG ：虽然以提升performance为目的，但是已经不是最优化问题了，而是在提升（降低）某概率

3. Gradient -> Objective
   - Policy gradients
   $$
    \hat{g}=\hat{\mathbb{E}}_{t}\left[\nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) \hat{A}_{t}\right]
   $$
   - 根据梯度，反向构造优化目标；Can differentiate the following loss
   $$
    L^{P G}(\theta)=\hat{\mathbb{E}}_{t}\left[\log \pi_{\theta}\left(a_{t} \mid s_{t}\right) \hat{A}_{t}\right]
   $$
   - Equivalently differentiate（不想更新太多,同时 diff log 转化）
   $$
    L_{\theta_{\mathrm{old}}}^{I S}(\theta)=\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t}     \mid s_{t}\right)}{\pi_{\theta_{\mathrm{old}}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}\right]
   $$
    at $\theta=\theta_{\text {old }}$, state-actions are sampled using $\theta_{\text {old. }}(I S=$ importance sampling) 
    Just the chain rule: $\left.\nabla_{\theta} \log f(\theta)\right|_{\theta_{\text {old }}}=\frac{\left.\nabla_{\theta} f(\theta)\right|_{\theta_{\text {old }}}}{f\left(\theta_{\text {old }}\right)}=\left.\nabla_{\theta}\left(\frac{f(\theta)}{f\left(\theta_{\text {old }}\right)}\right)\right|_{\theta_{\text {old }}}$

4. 换一种说法：Surrogate Loss : Importance Sampling
   - $\mathbb{E}_{s_{t} \sim \pi_{\theta_{\text {old }}}, a_{t} \sim \pi_{\theta}}\left[A^{\pi}\left(s_{t}, a_{t}\right)\right]$\
    $=\mathbb{E}_{s_{t} \sim \pi_{\theta_{\text {old }}, a_{t} \sim \pi_{\theta_{\text {old }}}}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)} A^{\pi_{\theta_{\text {old }}}}\left(s_{t}, a_{t}\right)\right]}$\
     $=\mathbb{E}_{s_{t} \sim \pi_{\theta_{\text {old }}}, a_{t} \sim \pi_{\theta_{\text {old }}}}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}\right]$\
     $=L_{\theta_{\text {old }}}^{I S}(\theta)$
   - 个人理解：trajectory 是由 old policy 收集的，但是更新的是new policy，所以在期望表达是需要使用 importance sampling

5. Trust Region Policy Optimization：用KL约束幅度
   - Define the following trust region update:
   $$
    \begin{array}{ll}
    \underset{\theta}{\operatorname{maximize}} & \hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}\right] \\
    \text { subject to } & \hat{\mathbb{E}}_{t}\left[\mathrm{KL}\left[\pi_{\theta_{\mathrm{old}}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right] \leq \delta
    \end{array}
   $$
   - Also worth considering using a penalty instead of a constraint
   $$
    \underset{\theta}{\operatorname{maximize}} \quad \hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}\right]-\beta \hat{\mathbb{E}}_{t}\left[\mathrm{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right]
   $$

【针对TRPO,可以使用局部线性化、quadra化】
- Suggested optimizing surrogate loss $L^{P G}$ or $L^{I S}$
- Suggested using $\mathrm{KL}$ to constrain size of update
- Corresponds to natural gradient step $F^{-1} g$ under linear quadratic approximation
- Can solve for this step approximately using conjugate gradient method

【TRPO 和其他方法的关系】
- Linear-quadratic approximation $+$ penalty $\Rightarrow$ natural gradient
- No constraint $\Rightarrow$ policy iteration
- Euclidean penalty instead of $\mathrm{KL} \Rightarrow$ vanilla policy gradient

# 简化改进版：Proximal Policy Optimization: Clipping Objective
1. 核心改进
- Recall the surrogate objective
$$
L^{I S}(\theta)=\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}\right]=\hat{\mathbb{E}}_{t}\left[r_{t}(\theta) \hat{A}_{t}\right]
$$
- Form a lower bound via clipped importance ratios
$$
L^{C L I P}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right]
$$

2. Pseudo Code
   - for iteration $=1,2, \ldots$ do
     - Run policy for $T$ timesteps or $N$ trajectories
     - Estimate advantage function at all timesteps
     - Do SGD on $L^{C L I P}(\theta)$ objective for some number of epochs end for
- A bit better than TRPO on continuous control, much better on Atari
- Compatible with multi-output networks and RNNs




# Numerical Problem and Implementation

1. 原始的estimate of gradient：$\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right) \hat{Q}_{i, t}$
   1. 可以理解为，计算多个gradient然后相加，这也太慢了，因此尝试构造一个方便 pytorch 梯度下降的对象
   2. "pseudo-loss" as a weighted maximum likelihood:
      1. $\nabla_{\theta} J_{\mathrm{ML}}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right) \quad J_{\mathrm{ML}}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log \pi_{\theta}\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right)$


2. 原始的 gradient 可能会发生步长不一致的缺点
   1. 缺点描述：$\theta^{\prime} \leftarrow \arg \max _{\theta_{\prime}}\left(\theta^{\prime}-\theta\right)^{T} \nabla_{\theta} J(\theta)$ s.t. $\left\|\theta^{\prime}-\theta\right\|^{2} \leq \epsilon$
      1. 尽管参数的更新幅度被约束了，但是可能发生比如 2 个参数，一个更新幅度大，另一个更新幅度小
      2. 上述情况会发生什么后果呢？
         - 在某个方向上下降的很快，但是它占据了主要更新的主要量，导致另一个弱势参数方向变化始终不明显，最终函数的结果容易陷入一个次优但是提升很难继续提升的处境。

   2. Nature Gradient： $\theta^{\prime} \leftarrow \arg \max _{\theta^{\prime}}\left(\theta^{\prime}-\theta\right)^{T} \nabla_{\theta} J(\theta)$ s.t. $D\left(\pi_{\theta^{\prime}}, \pi_{\theta}\right) \leq \epsilon$
      1. $D_{\mathrm{KL}}\left(\pi_{\theta^{\prime}} \| \pi_{\theta}\right)=E_{\pi_{\theta^{\prime}}}\left[\log \pi_{\theta}-\log \pi_{\theta^{\prime}}\right]$
      2. $D_{\mathrm{KL}}\left(\pi_{\theta^{\prime}} \| \pi_{\theta}\right) \approx\left(\theta^{\prime}-\theta\right)^{T} \mathbf{F}\left(\theta^{\prime}-\theta\right) \quad$
      3. Fisher Information $\mathbf{F}=E_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(\mathbf{a} \mid \mathbf{s}) \nabla_{\theta} \log \pi_{\theta}(\mathbf{a} \mid \mathbf{s})^{T}\right]$
      4. $\theta \leftarrow \theta+\alpha \mathbf{F}^{-1} \nabla_{\theta} J(\theta)$


   3. 上述的具体细节可以参考：Natrue Gradient，TRPO，conjugate gradients