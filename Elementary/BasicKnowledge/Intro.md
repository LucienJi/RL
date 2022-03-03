0. RL 和 Supervised learning 的区别在于：
   1. there is a feedback cycle; the next decision、action is in the situation that is the consequence of what you did before.

1. Markov Decision Procoss:\
   An MDP is defined by:
   - Set of states $S$
   - Set of actions $A$
   - Transition function $P\left(s^{\prime} \mid s, a\right)$ , 可以是 deterministic，也可以是 stochastic
   - Reward function $R\left(s, a, s^{\prime}\right)$
   - Start state $s_{0}$
   - Discount factor $\gamma$
   - Horizon $H$ , 可以是 infinity

2. Goal: $\max _{\pi} \mathrm{E}\left[\sum_{t=0}^{H} \gamma^{t} R\left(S_{t}, A_{t}, S_{t+1}\right) \mid \pi\right]$
   1. Optimal Control: given an MDP, find the optimal policy

3. Policy 分类:
   1. Deterministic Policy: mapping from a state to an action
   2. Stochastic Policy: mapping from a state a distribution over action

[理论证明通常实现于有限维度下，在连续动作空间，连续状态空间下可以参考相同的思想]

4. Exact Methods：
   1. Value Iteration: 
      1. Goal:  $V^{*}(s)=\max _{\pi} \mathbb{E}\left[\sum_{t=0}^{H} \gamma^{t} R\left(s_{t}, a_{t}, s_{t+1}\right) \mid \pi, s_{0}=s\right]$
         1. sum of discounted rewards when starLng from state s and acLng opLmally
      2. How to iterate:
         1. $V_{0}^{*}(s)=$ optimal value for state s when $\mathrm{H}=0$, $V_{0}^{*}(s)=0 \quad \forall s$
         2. $V_{k}^{*}(s)=\max _{a} \sum_{s^{\prime}} P\left(s^{\prime} \mid s, a\right)\left(R\left(s, a, s^{\prime}\right)+\gamma V_{k-1}^{*}\left(s^{\prime}\right)\right)$
         3. *Bellman Update*:  Algorithm
            - Start with $V_{0}^{*}(s)=0$ for all $\mathrm{s}$
            - For $\mathrm{k}=1, \ldots, \mathrm{H}:$
            - For all states $s$ in S:

            - $\begin{aligned}
                &V_{k}^{*}(s) \leftarrow \max _{a} \sum_{s^{\prime}} P\left(s^{\prime} \mid s, a\right)\left(R\left(s, a, s^{\prime}\right)+\gamma V_{k-1}^{*}\left(s^{\prime}\right)\right) \\
                &\pi_{k}^{*}(s) \leftarrow \arg \max _{a} \sum_{s^{\prime}} P\left(s^{\prime} \mid s, a\right)\left(R\left(s, a, s^{\prime}\right)+\gamma V_{k-1}^{*}\left(s^{\prime}\right)\right)
                \end{aligned}$
        1. Theorem: Value Iteration Converges
           1. 关注一点，用 H 的角度去思考，全局一次更新 == H=1 时最优的结果
           2. Contraction: $\left\|V_{i+1}-V_{i}\right\|<\epsilon, \Rightarrow\left\|V_{i+1}-V^{*}\right\|<2 \epsilon \gamma /(1-\gamma)$


   2. Policy Iteration
      1. Q-values: expected utility starting in s, taking action a, and (thereafter)acting optimally
         1. Bellman Equation(此处是最优Bellman): $Q^{*}(s, a)=\sum_{s^{\prime}} P\left(s^{\prime} \mid s, a\right)\left(R\left(s, a, s^{\prime}\right)+\gamma \max _{a^{\prime}} Q^{*}\left(s^{\prime}, a^{\prime}\right)\right)$
         2. Iteration: $Q_{k+1}^{*}(s, a) \leftarrow \sum_{s^{\prime}} P\left(s^{\prime} \mid s, a\right)\left(R\left(s, a, s^{\prime}\right)+\gamma \max _{a^{\prime}} Q_{k}^{*}\left(s^{\prime}, a^{\prime}\right)\right)$
      2. Policy Evaluation
         1. For a given policy $\pi(s)$, 注意点，此处反映的是这个策略下的价值函数，并不是最优情况下value
            1. Deterministic Policy： $V_{k}^{\pi}(s) \leftarrow \sum_{s^{\prime}} P\left(s^{\prime} \mid s, \pi(s)\right)\left(R\left(s, \pi(s), s^{\prime}\right)+\gamma V_{k-1}^{\pi}(s)\right)$;  注意点，此处不同于 iteration ，前者用的是 max，这里是别无选择的
            2. Stochastic Policy: $V_{k+1}^{\pi}(s) \leftarrow \sum_{s^{\prime}} \sum_{a} \pi(a \mid s) P\left(s^{\prime} \mid s, a\right)\left(R\left(s, a, s^{\prime}\right)+\gamma V_{k}^{\pi}\left(s^{\prime}\right)\right)$
      3. Policy Iteration:
         1. Brief, 根据 policy 更新 value；根据value，更新贪婪policy
         2. - Policy evaluation for current policy $\pi_{k}$ :
            - Iterate until convergence
              - $
              V_{i+1}^{\pi_{k}}(s) \leftarrow \sum_{s^{\prime}} P\left(s^{\prime} \mid s, \pi_{k}(s)\right)\left[R\left(s, \pi(s), s^{\prime}\right)+\gamma V_{i}^{\pi_{k}}\left(s^{\prime}\right)\right]
              $

            - Policy improvement: find the best action according to one-step look-ahead
              - $
              \pi_{k+1}(s) \leftarrow \underset{a}{\arg \max } \sum_{s^{\prime}} P\left(s^{\prime} \mid s, a\right)\left[R\left(s, a, s^{\prime}\right)+\gamma V^{\pi_{k}}\left(s^{\prime}\right)\right]
              $

           1. 关注点：
              1. 更新用的是箭头，Bellman Equation 是 等号，若满足等号，其实是满足了稳定的定义


5. Sampling-based Method：(Tabular) Q-learning
   1. Q-Learning
      - Q-value iteration: $Q_{k+1}(s, a) \leftarrow \sum_{s^{\prime}} P\left(s^{\prime} \mid s, a\right)\left(R\left(s, a, s^{\prime}\right)+\gamma \max _{a^{\prime}} Q_{k}\left(s^{\prime}, a^{\prime}\right)\right)$
      - Rewrite as expectation: $Q_{k+1} \leftarrow \mathbb{E}_{s^{\prime} \sim P\left(s^{\prime} \mid s, a\right)}\left[R\left(s, a, s^{\prime}\right)+\gamma \max _{a^{\prime}} Q_{k}\left(s^{\prime}, a^{\prime}\right)\right]$
      - 
   2. (Tabular) Q-Learning: replace expectation by samples
      - For an state-action pair $(s, a)$, receive: $s^{\prime} \sim P\left(s^{\prime} \mid s, a\right)$
      - Consider your old estimate: $Q_{k}(s, a)$
      - Consider your new sample estimate:
        $$
        \operatorname{target}\left(s^{\prime}\right)=R\left(s, a, s^{\prime}\right)+\gamma \underset{a^{\prime}}{\max } Q_{k}\left(s^{\prime}, a^{\prime}\right)
        $$
      - Incorporate the new estimate into a running average:
        $$
        Q_{k+1}(s, a) \leftarrow(1-\alpha) Q_{k}(s, a)+\alpha\left[\operatorname{target}\left(s^{\prime}\right)\right]
        $$
   
    3. Q-learning Properties:
       1. Q-learning converges to optimal policy -- even if you’re acting suboptimally! So this is off-policy method.(我们并不需要好的策略，纯随机也能使Q-learning converge)


6. Sampling - based Method:(Tabular) TD - Learning
   1. Tabular Based 的方法，不太能实现 policy improvement，因为需要 argmax，同理，value iteration 也需要 max next state value，因为不能假设sample next state
   2. Policy evaluation for current policy $\pi_{k}:$
      - Iterate until convergence
        $$
        V_{i+1}^{\pi_{k}}(s) \leftarrow \mathbb{E}_{s^{\prime} \sim P\left(s^{\prime} \mid s, \pi_{k}(s)\right)}\left[R\left(s, \pi_{k}(s), s^{\prime}\right)+\gamma V_{i}^{\pi_{k}}\left(s^{\prime}\right)\right]
        $$


7. Sampling - based Method：（Approximate） Q - Learning
   1. Instead of a table, we have a parametrized Q function: $Q_{\theta}(s, a)$
      - Can be a linear function in features:
        $$
        Q_{\theta}(s, a)=\theta_{0} f_{0}(s, a)+\theta_{1} f_{1}(s, a)+\cdots+\theta_{n} f_{n}(s, a)
        $$
      - Or a complicated neural net
      - Update:
        $$
        \theta_{k+1} \leftarrow \theta_{k}-\left.\alpha \nabla_{\theta}\left[\frac{1}{2}\left(Q_{\theta}(s, a)-\operatorname{target}\left(s^{\prime}\right)\right)^{2}\right]\right|_{\theta=\theta_{k}}
        $$