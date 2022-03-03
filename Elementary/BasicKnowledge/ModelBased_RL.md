
# Model Based Planning

1. Model-based Planning
   1. How can we choose the action under perfect knowledge of system dynamic ? optimal control,trajectory optimization,planning
   2. Loop：
      1. Closed Loop：看 s，给 a，接受 s',无限循环. 大部分RL 的解决方案
      2. Open Loop：看 s，给 a1,a2,a3,,,,,,
   3. Objective: 
      - Deterministic:  $\mathbf{a}_{1}, \ldots, \mathbf{a}_{T}=\arg \max _{\mathbf{a}_{1}, \ldots, \mathbf{a}_{T}} \sum_{t=1}^{T} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)$ s.t. $\mathbf{s}_{t+1}=f\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)$
      - Stochastic: 
        - $p_{\theta}\left(\mathbf{s}_{1}, \ldots, \mathbf{s}_{T} \mid \mathbf{a}_{1}, \ldots, \mathbf{a}_{T}\right)=p\left(\mathbf{s}_{1}\right) \prod_{t=1}^{T} p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}\right)$
        - 但是，这样选择 动作并不合理：$\mathbf{a}_{1}, \ldots, \mathbf{a}_{T}=\arg \max _{\mathbf{a}_{1}, \ldots, \mathbf{a}_{T}} E\left[\sum_{t} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right) \mid \mathbf{a}_{1}, \ldots, \mathbf{a}_{T}\right]$. 稍微解释一下为什么 open loop 会不合理：假设是在 dynamic model 是stochastic的，那么以考试为任务：选择是否参与考试；考对 + 1分；考错 -1分，事实上应该根据考试的难易来选择是否参加考试。

   4. 简单的 Model based 算法
      1. Stochastic Optimization：
         1. $\mathbf{a}_{1}, \ldots, \mathbf{a}_{T}=\arg \max _{\mathbf{a}_{1}, \ldots, \mathbf{a}_{T}} J\left(\mathbf{a}_{1}, \ldots, \mathbf{a}_{T}\right)$， like $\mathbf{A}=\arg \max _{\mathbf{A}} J(\mathbf{A})$
         2. pick $\mathbf{A}_{1}, \ldots, \mathbf{A}_{N}$ from some distribution (e.g., uniform)
         3. choose $\mathbf{A}_{i}$ based on $\arg \max _{i} J\left(\mathbf{A}_{i}\right)$
      2. Cross-Entropy Method(CEM)
         1. sample $\mathbf{A}_{1}, \ldots, \mathbf{A}_{N}$ from $p(\mathbf{A})$
         2. evaluate $J\left(\mathbf{A}_{1}\right), \ldots, J\left(\mathbf{A}_{N}\right)$
         3. pick the elites $\mathbf{A}_{i_{1}}, \ldots, \mathbf{A}_{i_{M}}$ with the highest value, where $M<N$
         4. refit $p(\mathbf{A})$ to the elites $\mathbf{A}_{i_{1}}, \ldots, \mathbf{A}_{i_{M}}$
      3. Monte Carlo Tree Search(MCTS)
         1. 如何展开，以及何时展开？
            1. find a leaf $s_{l}$ using TreePolicy $\left(s_{1}\right)$
            2. evaluate the leaf using DefaultPolicy $\left(s_{l}\right)$
            3. update all values in tree between $s_{1}$ and $s_{l}$，take best a from $s_{1}$
            4. UCT TreePolicy $\left(s_{t}\right)$
               - if $s_{t}$ not fully expanded, choose new $a_{t}$ else choose child with best $\operatorname{Score}\left(s_{t+1}\right)$
               - $\operatorname{Score}\left(s_{t}\right)=\frac{Q\left(s_{t}\right)}{N\left(s_{t}\right)}+2 C \sqrt{\frac{2 \ln N\left(s_{t-1}\right)}{N\left(s_{t}\right)}}$, Evaluation + Rarety

## Linear Case: LQR
1. Objectif: $\min _{\mathbf{u}_{1}, \ldots, \mathbf{u}_{T}, \mathbf{x}_{1}, \ldots, \mathbf{x}_{T}} \sum_{t=1}^{T} c\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right)$ s.t. $\mathbf{x}_{t}=f\left(\mathbf{x}_{t-1}, \mathbf{u}_{t-1}\right)$
2. Linear Forward Model:  $f\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right)=\mathbf{F}_{t}\left[\begin{array}{l}\mathbf{x}_{t} \\ \mathbf{u}_{t}\end{array}\right]+\mathbf{f}_{t}$
3. Quadratic Cost Model:  $c\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right)=\frac{1}{2}\left[\begin{array}{l}\mathbf{x}_{t} \\ \mathbf{u}_{t}\end{array}\right]^{T} \mathbf{C}_{t}\left[\begin{array}{l}\mathbf{x}_{t} \\ \mathbf{u}_{t}\end{array}\right]+\left[\begin{array}{l}\mathbf{x}_{t} \\ \mathbf{u}_{t}\end{array}\right]^{T} \mathbf{c}_{t}$
4. 核心分析方法：
   1. 先看末状态，解出在已知末状态下，末控制的表达式。为什么？因为利用recursive的方法，末状态是可以通过forward model 推出来的。
   2. Base case: solve for $\mathbf{u}_{T}$ only
      - $$
        Q\left(\mathbf{x}_{T}, \mathbf{u}_{T}\right)=\operatorname{const}+\frac{1}{2}\left[\begin{array}{l}
        \mathbf{x}_{T} \\
        \mathbf{u}_{T}
        \end{array}\right]^{T} \mathbf{C}_{T}\left[\begin{array}{l}
        \mathbf{x}_{T} \\
        \mathbf{u}_{T}
        \end{array}\right]+\left[\begin{array}{c}
        \mathbf{x}_{T} \\
        \mathbf{u}_{T}
        \end{array}\right]^{T} \mathbf{c}_{T}
        $$
      - $\nabla_{\mathbf{u}_{T}} Q\left(\mathbf{x}_{T}, \mathbf{u}_{T}\right)=\mathbf{C}_{\mathbf{u}_{T}, \mathbf{x}_{T}} \mathbf{x}_{T}+\mathbf{C}_{\mathbf{u}_{T}, \mathbf{u}_{T}} \mathbf{u}_{T}+\mathbf{c}_{\mathbf{u}_{T}}^{T}=0$
      - 解开来： $\mathbf{u}_{T}=-\mathbf{C}_{\mathbf{u}_{T}, \mathbf{u}_{T}}^{-1}\left(\mathbf{C}_{\mathbf{u}_{T}, \mathbf{x}_{T}} \mathbf{x}_{T}+\mathbf{c}_{\mathbf{u}_{T}}\right) \quad \mathbf{u}_{T}=\mathbf{K}_{T} \mathbf{x}_{T}+\mathbf{k}_{T}$
   3. 因为控制是被状态完全决定的，所以可以把cost函数中的控制替换为状态
      - $V\left(\mathbf{x}_{T}\right)=$ const $+\frac{1}{2}\left[\begin{array}{c}\mathbf{x}_{T} \\ \mathbf{K}_{T} \mathbf{x}_{T}+\mathbf{k}_{T}\end{array}\right]^{T} \mathbf{C}_{T}\left[\begin{array}{c}\mathbf{x}_{T} \\ \mathbf{K}_{T} \mathbf{x}_{T}+\mathbf{k}_{T}\end{array}\right]+\left[\begin{array}{c}\mathbf{x}_{T} \\ \mathbf{K}_{T} \mathbf{x}_{T}+\mathbf{k}_{T}\end{array}\right]^{T} \mathbf{c}_{T}$
      - $V\left(\mathbf{x}_{T}\right)=$ const $+\frac{1}{2} \mathbf{x}_{T}^{T} \mathbf{V}_{T} \mathbf{x}_{T}+\mathbf{x}_{T}^{T} \mathbf{v}_{T}$
      - $\mathbf{V}_{T}=\mathbf{C}_{\mathbf{x}_{T}, \mathbf{x}_{T}}+\mathbf{C}_{\mathbf{x}_{T}, \mathbf{u}_{T}} \mathbf{K}_{T}+\mathbf{K}_{T}^{T} \mathbf{C}_{\mathbf{u}_{T}, \mathbf{x}_{T}}+\mathbf{K}_{T}^{T} \mathbf{C}_{\mathbf{u}_{T}, \mathbf{u}_{T}} \mathbf{K}_{T}$
      - $\mathbf{v}_{T}=\mathbf{c}_{\mathbf{x}_{T}}+\mathbf{C}_{\mathbf{x}_{T}, \mathbf{u}_{T}} \mathbf{k}_{T}+\mathbf{K}_{T}^{T} \mathbf{C}_{\mathbf{u}_{T}}+\mathbf{K}_{T}^{T} \mathbf{C}_{\mathbf{u}_{T}, \mathbf{u}_{T}} \mathbf{k}_{T}$

   4. 如何递归呢？因为 $u_{t-1}$ 会决定 $x_{t}$,重新写Q函数，用forward function代替$x_{t}$
      - $Q\left(\mathbf{x}_{T-1}, \mathbf{u}_{T-1}\right)=\operatorname{const}+\frac{1}{2}\left[\begin{array}{l}\mathbf{x}_{T-1} \\ \mathbf{u}_{T-1}\end{array}\right]^{T} \mathbf{C}_{T-1}\left[\begin{array}{l}\mathbf{x}_{T-1} \\ \mathbf{u}_{T-1}\end{array}\right]+\left[\begin{array}{l}\mathbf{x}_{T-1} \\ \mathbf{u}_{T-1}\end{array}\right]^{T} \mathbf{c}_{T-1}+V\left(f\left(\mathbf{x}_{T-1}, \mathbf{u}_{T-1}\right)\right)$
      - 你看，又可以重新写成 quadratic + linear 的形式，重复上述步骤$Q\left(\mathbf{x}_{T-1}, \mathbf{u}_{T-1}\right)=$ const $+\frac{1}{2}\left[\begin{array}{l}\mathbf{x}_{T-1} \\ \mathbf{u}_{T-1}\end{array}\right]^{T} \mathbf{Q}_{T-1}\left[\begin{array}{l}\mathbf{x}_{T-1} \\ \mathbf{u}_{T-1}\end{array}\right]+\left[\begin{array}{l}\mathbf{x}_{T-1} \\ \mathbf{u}_{T-1}\end{array}\right]^{T} \mathbf{q}_{T-1}$
      - $\mathbf{Q}_{T-1}=\mathbf{C}_{T-1}+\mathbf{F}_{T-1}^{T} \mathbf{V}_{T} \mathbf{F}_{T-1}$
      - $\mathbf{q}_{T-1}=\mathbf{c}_{T-1}+\mathbf{F}_{T-1}^{T} \mathbf{V}_{T} \mathbf{f}_{T-1}+\mathbf{F}_{T-1}^{T} \mathbf{v}_{T}$
5. Linear Case: LQR
   - Backward recursion
     - for $t=T$ to 1 :
     -  $$
        \begin{aligned}
        &\mathbf{Q}_{t}=\mathbf{C}_{t}+\mathbf{F}_{t}^{T} \mathbf{V}_{t+1} \mathbf{F}_{t} \\
        &\mathbf{q}_{t}=\mathbf{c}_{t}+\mathbf{F}_{t}^{T} \mathbf{V}_{t+1} \mathbf{f}_{t}+\mathbf{F}_{t}^{T} \mathbf{v}_{t+1} \\
        &Q\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right)=\text { const }+\frac{1}{2}\left[\begin{array}{c}
        \mathbf{x}_{t} \\
        \mathbf{u}_{t}
        \end{array}\right]^{T} \mathbf{Q}_{t}\left[\begin{array}{c}
        \mathbf{x}_{t} \\
        \mathbf{u}_{t}
        \end{array}\right]+\left[\begin{array}{l}
        \mathbf{x}_{t} \\
        \mathbf{u}_{t}
        \end{array}\right]^{T} \mathbf{q}_{t} \\
        &\mathbf{u}_{t} \leftarrow \arg \min _{\mathbf{u}_{t}} Q\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right)=\mathbf{K}_{t} \mathbf{x}_{t}+\mathbf{k}_{t} \\
        &\mathbf{K}_{t}=-\mathbf{Q}_{\mathbf{u}_{t}, \mathbf{u}_{t}}^{-1} \mathbf{Q}_{\mathbf{u}_{t}, \mathbf{x}_{t}} \\
        &\mathbf{k}_{t}=-\mathbf{Q}_{\mathbf{u}_{t}, \mathbf{u}_{t}}^{-1} \mathbf{q}_{\mathbf{u}_{t}} \\
        &\mathbf{V}_{t}=\mathbf{Q}_{\mathbf{x}_{t}, \mathbf{x}_{t}}+\mathbf{Q}_{\mathbf{x}_{t}, \mathbf{u}_{t}} \mathbf{K}_{t}+\mathbf{K}_{t}^{T} \mathbf{Q}_{\mathbf{u}_{t}, \mathbf{x}_{t}}+\mathbf{K}_{t}^{T} \mathbf{Q}_{\mathbf{u}_{t}, \mathbf{u}_{t}} \mathbf{K}_{t} \\
        \mathbf{v}_{t}=\mathbf{q}_{\mathbf{x}_{t}}+\mathbf{Q}_{\mathbf{x}_{t}, \mathbf{u}_{t}} \mathbf{k}_{t}+\mathbf{K}_{t}^{T} \mathbf{Q}_{\mathbf{u}_{t}}+\mathbf{K}_{t}^{T} \mathbf{Q}_{\mathbf{u}_{t}, \mathbf{u}_{t}} \mathbf{k}_{t} \\
        &V\left(\mathbf{x}_{t}\right)=\text { const }+\frac{1}{2} \mathbf{x}_{t}^{T} \mathbf{V}_{t} \mathbf{x}_{t}+\mathbf{x}_{t}^{T} \mathbf{v}_{t}
        \end{aligned}
        $$
   - Forward recursion
     - for $t=1$ to $T:$
     - $$
        \begin{aligned}
        &\mathbf{u}_{t}=\mathbf{K}_{t} \mathbf{x}_{t}+\mathbf{k}_{t} \\
        &\mathbf{x}_{t+1}=f\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right)
        \end{aligned}
       $$

## Stochastic Case
1. 最大的改变
   - $f\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right)=\mathbf{F}_{t}\left[\begin{array}{l}\mathbf{x}_{t} \\ \mathbf{u}_{t}\end{array}\right]+\mathbf{f}_{t}$
   - $\mathbf{x}_{t+1} \sim p\left(\mathbf{x}_{t+1} \mid \mathbf{x}_{t}, \mathbf{u}_{t}\right)$
   - $p\left(\mathbf{x}_{t+1} \mid \mathbf{x}_{t}, \mathbf{u}_{t}\right)=\mathcal{N}\left(\mathbf{F}_{t}\left[\begin{array}{l}\mathbf{x}_{t} \\ \mathbf{u}_{t}\end{array}\right]+\mathbf{f}_{t}, \Sigma_{t}\right)$

2. Solution: choose actions according to $\mathbf{u}_{t}=\mathbf{K}_{t} \mathbf{x}_{t}+\mathbf{k}_{t}$
3. 但是，不再是 Open-Loop，而是转变为 Close - Loop，类似 Model Predictive Control，需要重新plan + execute，然后看最终到达哪个state


## Non-Linear Model: DDP/iterative LQR
1. 用Taylor来 approximate Linear,quadratic
   - $f\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right) \approx f\left(\hat{\mathbf{x}}_{t}, \hat{\mathbf{u}}_{t}\right)+\nabla_{\mathbf{x}_{t}, \mathbf{u}_{t}} f\left(\hat{\mathbf{x}}_{t}, \hat{\mathbf{u}}_{t}\right)\left[\begin{array}{l}\mathbf{x}_{t}-\hat{\mathbf{x}}_{t} \\ \mathbf{u}_{t}-\hat{\mathbf{u}}_{t}\end{array}\right]$
   - $c\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right) \approx c\left(\hat{\mathbf{x}}_{t}, \hat{\mathbf{u}}_{t}\right)+\nabla_{\mathbf{x}_{t}, \mathbf{u}_{t}} c\left(\hat{\mathbf{x}}_{t}, \hat{\mathbf{u}}_{t}\right)\left[\begin{array}{c}\mathbf{x}_{t}-\hat{\mathbf{x}}_{t} \\ \mathbf{u}_{t}-\hat{\mathbf{u}}_{t}\end{array}\right]+\frac{1}{2}\left[\begin{array}{c}\mathbf{x}_{t}-\hat{\mathbf{x}}_{t} \\ \mathbf{u}_{t}-\hat{\mathbf{u}}_{t}\end{array}\right]^{T} \nabla_{\mathbf{x}_{t}, \mathbf{u}_{t}}^{2} c\left(\hat{\mathbf{x}}_{t}, \hat{\mathbf{u}}_{t}\right)\left[\begin{array}{c}\mathbf{x}_{t}-\hat{\mathbf{x}}_{t} \\ \mathbf{u}_{t}-\hat{\mathbf{u}}_{t}\end{array}\right]$

2. PseudoCode
   - Iterative LQR (simplified pseudocode)
     - until convergence:
     - $$
        \begin{aligned}
        &\mathbf{F}_{t}=\nabla_{\mathbf{x}_{t}, \mathbf{u}_{t}} f\left(\hat{\mathbf{x}}_{t}, \hat{\mathbf{u}}_{t}\right) \\
        &\mathbf{c}_{t}=\nabla_{\mathbf{x}_{t}, \mathbf{u}_{t}} c\left(\hat{\mathbf{x}}_{t}, \hat{\mathbf{u}}_{t}\right) \\
        &\mathbf{C}_{t}=\nabla_{\mathbf{x}_{t}, \mathbf{u}_{t}}^{2} c\left(\hat{\mathbf{x}}_{t}, \hat{\mathbf{u}}_{t}\right)
        \end{aligned}
       $$
     - Run LQR backward pass on state $\delta \mathbf{x}_{t}=\mathbf{x}_{t}-\hat{\mathbf{x}}_{t}$ and action $\delta \mathbf{u}_{t}=\mathbf{u}_{t}-\hat{\mathbf{u}}_{t}$ Run forward pass with real nonlinear dynamics and $\mathbf{u}_{t}=\mathbf{K}_{t}\left(\mathbf{x}_{t}-\hat{\mathbf{x}}_{t}\right)+\mathbf{k}_{t}+\hat{\mathbf{u}}_{t}$ Update $\hat{\mathbf{x}}_{t}$ and $\hat{\mathbf{u}}_{t}$ based on states and actions in forward pass


# Model Based RL

## Naive Approach
1. Why Model-Based:
   1. Sample - efficiency:
      1. gradient-free < fully online < Policy Gradient < Replay Buffer + Value Estimation < Model Based
   2. transferability & generality
   3. 假如是不太复杂的model，就可以使用LQR等强力算法


2. Model-Based 流程
   1. generate sample using policy -> fit the model -> optimize policy(model-based)
   2. Model-based optimization: 简单来说，在 t 时刻的受到的reward 不再是单个reward，而是根据model多次迭代后的累加reward。
   - 个人理解：
   - 原版：$\nabla_{\theta} U(\theta) \approx \hat{g}=\frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right) R\left(\tau^{(i)}\right)$
   - 原版，dynamic model 是梯度无关的 $\nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right) =\nabla_{\theta} \log [\prod_{t=0}^{H} \underbrace{P\left(s_{t+1}^{(i)} \mid s_{t}^{(i)}, u_{t}^{(i)}\right)}_{\text {dynamics model }} \cdot \underbrace{\pi_{\theta}\left(u_{t}^{(i)} \mid s_{t}^{(i)}\right)}_{\text {policy }}]$
   - 再版 model based：$s_{t+1} = f(s_{t},\pi_{\theta}(s_{t}))$,这样是不是可以回传了？


3. 原始的几种 model-based 和其优缺点
   1. Algorithm v0:
      - run base policy $\pi_{0}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)$ (e.g., random policy) to collect $\mathcal{D}=\left\{\left(\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime}\right)_{i}\right\}$
      - learn model $f_{\phi}(\mathbf{s}, \mathbf{a})$ to rminimize $\sum_{i}\left\|f_{\dot{\phi}}\left(\mathbf{s}_{i}, \mathbf{a}_{i}\right)-\mathbf{s}_{i}^{\prime}\right\|^{2}$
      - backpropagate through $f_{\phi}(\mathbf{s}, \mathbf{a})$ into policy to optimize $\pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)$

      - 缺点：fit model 时候用的 policy 和 updateing policy 时候的data是不一样的，distribution是不同的。通俗来说，fit model 可能sample 的是高速公路，结果在update policy 时候走得全是野路，model 根本没学过，回传效果更差。
      - 这个缺点，在 expressive model 上更加明显！！
   2. Algorithm v1:
      - run base policy $\pi_{0}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)$ (e.g., random policy) to collect $\mathcal{D}=\left\{\left(\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime}\right)_{i}\right\}$
      - learn model $f_{\phi}(\mathbf{s}, \mathbf{a})$ to rminimize $\sum_{i}\left\|f_{\dot{\varphi}}\left(\mathbf{s}_{i}, \mathbf{a}_{i}\right)-\mathbf{s}_{i}^{\prime}\right\|^{2}$
      - backpropagate through $f_{\dot{\infty}}(\mathbf{s}, \mathbf{a})$ into policy to optimize $\pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)$
      - run $\pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)$, appending visited tuples $\left(\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime}\right)$ to $\mathcal{D}$ **核心变动在这里，也就是不断renew model**


   3. Algorithm v2: (model predictive control,以前写机器人代码的时候用过)
      - run base policy $\pi_{0}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)$ (e.g., random policy) to collect $\mathcal{D}=\left\{\left(\mathbf{s} ; \mathbf{a}, \mathbf{s}^{\prime}\right)_{i}\right\}$
      - learn model $f_{\phi}(\mathbf{s}, \mathbf{a})$ to ruinimize $\sum_{i}\left\|f_{\phi}\left(\mathbf{s}_{i}, \mathbf{a}_{i}\right)-\mathbf{s}_{i}^{\prime}\right\|^{2}$
      - backpropagate through $f_{\phi}(\mathbf{s}, \mathbf{a})$ to choose actions.
      - exccute the first planned action, obscrve resulting state $\mathbf{s}^{\prime}$  **核心变动**，每次看很多步后的reward，但是实际上只采用第一步，以免中途走错路，然后错上加错。
      - append $\left(\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime}\right)$ to dataset $\mathcal{D}$
      - - 这个算法是有效地，但是 Computational Expensive
      - The more you replan, the less perfect each individual plan needs to be
      - Can use shorter horizons
4. 为什么说上述的三种 “backprop through dynamic model” 是naive的呢？
   - 累加reward，然后对这个planning后的累计reward做梯度，会导致不断回传的梯度增加，发生类似 RNN 中梯度爆炸、梯度消失的问题。同时也无法使用LSTM的技巧，因为 forward model 本身就未必简单可导，很多模型的forward model 就是真实的困难。

## 稍微改进一点的 Model-based approach
### Model-free optimization with a model
1. 原初版本：Dyna,online Q-learning algorithm that performs model-free RL with a model
2. 相同Idea 的众多衍生品：model-based acceleration,value expansion,policy optimization
3. 大致流程
   1. take some action $\mathbf{a}_{i}$ and observe $\left(\mathbf{s}_{i}, \mathbf{a}_{i}, \mathbf{s}_{i}^{\prime}, r_{i}\right)$, add it to $\mathcal{B}$
   2. sample mini-batch $\left\{\mathbf{s}_{j}, \mathbf{a}_{j}, \mathbf{s}_{j}^{\prime}, r_{j}\right\}$ from $\mathcal{B}$ uniformly
   3. use $\left\{\mathbf{s}_{j}, \mathbf{a}_{j}, \mathbf{s}_{j}^{\prime}\right\}$ to update model $\hat{p}\left(\mathbf{s}^{\prime} \mid \mathbf{s}, \mathbf{a}\right)$
   4. sample $\left\{\mathbf{s}_{j}\right\}$ from $\mathcal{B}$
   5. for each $\mathbf{s}_{j}$, perform model-based rollout with $\mathbf{a}=\pi(\mathbf{s})$
   6. use all transitions $\left(\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime}, r\right)$ along rollout to update Q-function
4. 优点：利用forward model 可以去往diverse states，某些情况下节约了simulation。同时短距离的rollout比长距离的rollout更加稳定。
5. 缺点：非常依赖forward model 的性能，假如没见过的state只能乱猜，那么policy只会在一些永远不会遇到的state中训练。
### Local models
1. Ideas: fit $\frac{d f}{d \mathbf{x}_{t}}, \frac{d f}{d \mathbf{u}_{t}}$
2. 具体流程：
   1. $\operatorname{run} p\left(\mathbf{u}_{t} \mid \mathbf{x}_{t}\right)$ on robot，collect $\mathcal{D}=\left\{\tau_{i}\right\}$
   2. fit dynamics $p\left(\mathbf{x}_{t+1} \mid \mathbf{x}_{t}, \mathbf{u}_{t}\right)$
      1. $p\left(\mathbf{x}_{t+1} \mid \mathbf{x}_{t}, \mathbf{u}_{t}\right)=\mathcal{N}\left(f\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right), \Sigma\right)$
      2. $f\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right) \approx \mathbf{A}_{t} \mathbf{x}_{t}+\mathbf{B}_{t} \mathbf{u}_{t}$
      3. $\mathbf{A}_{t}=\frac{d f}{d \mathbf{x}_{t}} \quad \mathbf{B}_{t}=\frac{d f}{d \mathbf{u}_{t}}$
   3. improve controller


3. 还有很多相关的组合 local policy 成为更好policy 的方法：
   1. guided policy search
   2. distillation 



## Uncertainty Aware for the model

1. 发现实验问题：Model-Based 的模型在训练初期的提升远不如 model-free 的模型
   1. 潜在原因是因为model是用 neural network 拟合的，所以在数据量小的时候，性能特别差，由它指导的 RL policy 更加不堪入目。


2. 两种 Uncertainty:
   1. aleatoric or statistical uncertainty: noise from sample, 即使拥有大量样本，噪音不可避免，但是不影响我们获得模型。
   2. epistemic or model uncertainty：来源于样本过少，我们其实并不知道模型是什么，此时也是 过拟合 最常见的时候。以走近悬崖这个任务为例，此时的 uncertainty 来源于还没有仔细的走过悬崖，所以不确定下一步会不会掉下去。


3. 如何表述 model uncertainty？ 小Idea
   1. 基本概念：
      1. Forward Model： based on neural network, $p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}\right)$, 输入 s,a,输出 s'
      2. $\arg \max _{\theta} \log p(\theta \mid \mathcal{D})$, 这可以被认为是（基于当前有限的样本）最好的权重。然后输出也变为： $\int p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}, \theta\right) p(\theta \mid \mathcal{D}) d \theta$
   2. Baysian Network
      1. 核心观点：权重不是 deterministic
      2. 方法1：粗糙近似：$p(\theta \mid \mathcal{D})=\prod_{i} p\left(\theta_{i} \mid \mathcal{D}\right)$
      3. 方法2：假设分布：$p\left(\theta_{i} \mid \mathcal{D}\right)=\mathcal{N}\left(\mu_{i}, \sigma_{i}\right)$, 方差也体现了 uncertainty
   3. Bootstrap ensembles
      1. $p(\theta \mid \mathcal{D}) \approx \frac{1}{N} \sum_{i} \delta\left(\theta_{i}\right)$
         - $\int p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}, \theta\right) p(\theta \mid \mathcal{D}) d \theta \approx \frac{1}{N} \sum_{i} p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}, \theta_{i}\right)$
      2. 注意点：对于 continuous action 来说，最终结果并不是一个 gaussian with mean of 所有输出mean的均值，而是多个gaussian的相加
      3. Training：generate independent datasets to get independent models。可以sample d from D with replacement。
      4. Random Initialization

4. Further Reading for Model-based RL
   - Deisenroth et al. PILCO: A Model-Based and Data-Efficient Approach to Policy Search.
   - Nagabandi et al. Neural Network Dynamics for ModelBased Deep Reinforcement Learning with Model-Free Fine-Tuning.
   - Chua et al. Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models.
   - Feinberg et al. Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning.
   - Buckman et al. Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion.


## Model-based with Images

1. 图片有什么难度？
   1. High dimension
   2. Redundancy
   3. Partial observation

2. 常见做法：Latent Space
   1. 数学角度：
      1. $p\left(\mathbf{o}_{t} \mid \mathbf{s}_{t}\right)$ observation model(注意，这是已知state，反推 obs，因为我们希望知道的是observation背后的真实状态，通常认为observation是由state 产出的)
      2. $p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}\right)$， dynamic model
      3. $p\left(r_{t} \mid \mathbf{s}_{t}, \mathbf{a}_{t}\right)$, reward model (因为通常认为，reward 是根据env的内部状态给的，而不是 observation)
      4. 数学目标：
         - 动态模型： $\max _{\phi} \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log p_{\phi}\left(\mathbf{s}_{t+1, i} \mid \mathbf{s}_{t, i}, \mathbf{a}_{t, i}\right)$
         - 隐层模型： $\max _{\phi} \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} E\left[\log p_{\phi}\left(\mathbf{s}_{t+1, i} \mid \mathbf{s}_{t, i}, \mathbf{a}_{t, i}\right)+\log p_{\phi}\left(\mathbf{o}_{t, i} \mid \mathbf{s}_{t, i}\right)\right]$，expectation w.r.t. $\left(\mathbf{s}_{t}, \mathbf{s}_{t+1}\right) \sim p\left(\mathbf{s}_{t}, \mathbf{s}_{t+1} \mid \mathbf{o}_{1: T}, \mathbf{a}_{1: T}\right)$ 
      5. 算法解决方案
         1. observation model 是不可知的，所以可以选择 用一个encoder来近似：
            1. Full smooth posterior: $q_{\psi}\left(\mathbf{s}_{t}, \mathbf{s}_{t+1} \mid \mathbf{o}_{1: T}, \mathbf{a}_{1: T}\right)$
            2. Single step encoder: $q_{\psi}\left(\mathbf{s}_{t} \mid \mathbf{o}_{t}\right)$

         2. Determinisitc or Stochastic ? 都可以，简单一点就 deterministic + single step：
            1. 用一个简单的网络就可以表示一个 encoder了： $q_{\psi}\left(\mathbf{s}_{t} \mid \mathbf{o}_{t}\right)=\delta\left(\mathbf{s}_{t}=g_{\psi}\left(\mathbf{o}_{t}\right)\right) \Rightarrow \mathbf{s}_{t}=g_{\psi}\left(\mathbf{o}_{t}\right)$
            2. 可以将目标函数转写为一个 tractable + fully differentiable： $\max _{\phi, \psi} \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log p_{\phi}\left(g_{\psi}\left(\mathbf{o}_{t+1, i}\right) \mid g_{\psi}\left(\mathbf{o}_{t, i}\right), \mathbf{a}_{t, i}\right)+\log p_{\phi}\left(\mathbf{o}_{t, i} \mid g_{\psi}\left(\mathbf{o}_{t, i}\right)\right) + \log p_{\phi}\left(r_{t, i} \mid g_{\psi}\left(\mathbf{o}_{t, i}\right)\right)$