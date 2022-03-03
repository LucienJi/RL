# Continuous Space
1. 数学基础
   1. set of all Probabilty Measure: $\mathscr{P}(\mathcal{X})$. [其中元素就是常说的密度函数，积分为1]
   2. set of all Measurable Funciton: $\mathscr{B}(\mathcal{X})$. [在认知中，几乎所有遇到的函数都是其中元素]
   3. $\rho \in \mathscr{P}(\mathcal{X})$，$f \in \mathscr{B}(\mathcal{X})$，将前者看为operator：$\rho f=\int_{\mathcal{X}} f(x) \rho(\mathrm{d} x)$. [这应该就是f在ρ下的期望了]

2. Markov Decision Processes
   1. 5 - tuple M = $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$, 用 measurable set 的角度去看。
   2. Transition Kernel: 从 pair 到probability measure of state: $P: \mathcal{S} \times \mathcal{A} \rightarrow \mathscr{P}(\mathcal{S})$
   3. Policy: 从 history 到 probability measure of action: $\pi_{t}: \mathcal{H}_{t} \rightarrow $\mathscr{P}(\mathcal{A})$
      1. Markovian: $\pi_{t}: \mathcal{S} \rightarrow \mathscr{P}(\mathcal{A})$
      2. Stationary: remove subscript t，不随时变化。
   4. State-action transition kernel induced by policy：$P^{\pi}: \mathcal{S} \times \mathcal{A} \rightarrow \mathscr{P}(\mathcal{S} \times \mathcal{A})$
      1. $\left(P^{\pi}\right)(\mathcal{B} \mid s, a)=\int_{\mathcal{S}} P\left(\mathrm{~d} s^{\prime} \mid s, a\right) \int_{\mathcal{A}} \pi\left(\mathrm{d} a^{\prime} \mid s^{\prime}\right) \delta_{\left(s^{\prime}, a^{\prime}\right)}(\mathcal{B})$
      2. 从state-action pair 到 probability measure of state-action pair。其中，B 是measurable set
3. RL setting
   1. Q funciton(based on policy):  $Q^{\pi}(s, a)=\mathbb{E}\left[\sum_{t=0}^{+\infty} \gamma^{t} R_{t} \mid S_{0}=s, A_{0}=a\right]$
      1. 此处期望在于：奖励分布，转移概率分布，策略的输出分布
   2. V function（based on policy）：$V^{\pi}(s)=\int_{\mathcal{A}} \pi(\mathrm{d} a \mid s) Q^{\pi}(s, a)$，相当于策略这个 probability measure 作为一个operator 作用在了Q上。

4. Bellman Operator
   1. Bellman Expectation Operator：可以理解用于为检验当前策略，可以用于更新 Q，V
      1. $T^{\pi} = \mathscr{B}(\mathcal{S} \times \mathcal{A}) \rightarrow \mathscr{B}(\mathcal{S} \times \mathcal{A})$，从 measurable function 到 measurable function
      2. [此处存疑，待查证随机控制理论]$\left(T^{\pi} f\right)(s, a)=r(s, a)+\left(P^{\pi} f\right)(s, a)$
   2. Bellman Optimal Operator: 不在乎策略！！！只管选最好的，用于更新最优Q和V
      1. $T^{*}: \mathscr{B}(\mathcal{S} \times \mathcal{A}) \rightarrow \mathscr{B}(\mathcal{S} \times \mathcal{A})$
      2. $\left(T^{*} f\right)(s, a)=r(s, a)+\gamma \int_{\mathcal{S}} P\left(\mathrm{~d} s^{\prime} \mid s, a\right) \max _{a^{\prime} \in \mathcal{A}} f\left(s^{\prime}, a^{\prime}\right)$
   3. $T^{\pi}$ and $T^{*}$ are $\gamma$-contractions in $L_{\infty}$-norm, 所以我们有不动点，所以 Iteration 是可以收敛的。
      1. $T^{\pi} Q^{\pi}=Q^{\pi}$
      2. $T^{*} Q^{*}=Q^{*}$

