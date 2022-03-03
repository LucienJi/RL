1. Approching new problem
   1. New algorithm:
      1. run experiments quickly: eg 摇晃摆
      2. Do hyperparameter search
      3. Interpret and visualize learning process: state visitation, value function eg 摇晃摆
      4. Construct toy problems where your idea will be strongest and weakest,where you have a sense of what it should do
      5. Counterpoint: don’t overfit algorithm to contrived problem(别执着于 toy problem)
      6. Useful to have medium-sized problems that you’re intimately familiar with (可以有一个中等程度的测试模板)
   2. New task
      1. Provide good feature：
         1. 测试期，显然图片输入比不上 （x,y）
      2. Shape reward function:
         1. 测试期不要使用 sparse reward
   3. POMDP Design
      1. Visualize random policy: does it sometime exhibit desired behaviors ?
         - 首先要明白，假如随机动作中都没机会有一丝丝进步，你不可能让你的 agent 学到任何东西，因为初始policy也一定是纯随机的。
      2. Human Control : 强调图片类输入，假如有 预处理的步骤，一定要确保，处理后的图片人类自己也能理解。
      3. Plot time series for observation and reward
   4. Baselines:每个人都会使用baseline
      1. 不可能只用 defaut para
      2. 推荐使用：well tuned Q-learning + SARSA, PG 
   5. Reproduce the published results
      1. In tuning process, need huge number of samples: 调参也要有耐心


2. Ongoing Development and Tuning
   1. 假如你发现算法在任务上成功了？
      1. 改变超参数， if it is sensitive the hyperparameters, it does not work.
      2. Look for health indicators
         - VF fit quality
         - Policy entropy
         - Update size in output space and parameter space
         - Standard diagnostics for deep networks
   2. 假如真的算法在这个任务上挺不错的？
      1. 尝试多个不同任务
      2. 尝试同一个任务使用不同的random seed
      3. 很多时候，同一个任务、同一个seed下，RL 就变成了regression问题
   3. 简化模型：
      1. 提出重复功能的模块和trick
      2. 网络设计的时候尝试ablating


3. Tuning Strategies
   1. Whitening the data
      1. 假如 observation 的 range 是不确定的话，那么observation
         - Compute running estimate of mean and standard deviation
         - $x^{\prime}=\operatorname{clip}((x-\mu) / \sigma,-10,10)$
      2. 不要rescale reward，尤其是 mean reward，因为这会改变游戏设定
      3. 也可以standardalize prediction
   2. 重要参数：
      1. Discount $\gamma$ 通常代表最远的参考举例：Effective time horizon: $1+\gamma+\gamma^{2}+\cdots=1 /(1-\gamma)$
      2. In $\operatorname{TD}(\lambda)$ methods, can get away with high $\gamma$ when $\lambda<1$

   3. 关键 diagnostics
      1. 除了reward mean 以外，还可以看 reward max，min，std，因为假如 reward max 的确很优秀，那么说明 agent 至少已经探索到了目标路径，（假如模型正确）RL 模型会不断探索这条路，最终mean reward 会一同提升。

4. Policy Gradient 调参特辑
   1. Entropy:
      1. 不可以过早下降，因为这可能意味着 no learnin
      2. 使用 entropy bonus、KL penalty 可以缓解这个问题
   2. KL：  
      1. 计算 $\mathrm{KL}\left[\pi_{\mathrm{old}}(\cdot \mid s), \pi(\cdot \mid s)\right]$
      2. KL spike,快速上升
   3. Explained Variance
      - explained variance $=\frac{1-\text { Var[empirical return-predicted value] }}{\operatorname{Var}[\text { empirical return }]}$
      - $\underbrace{\sum_{i=1}^{N}\left(z_{i}-\bar{z}\right)^{2}}_{\text {TSS }}=\underbrace{\sum_{i=1}^{N}\left(z_{i}-\hat{z}_{i}\right)^{2}}_{\text {RSS }}+\underbrace{\sum_{i=1}^{N}\left(\hat{z}_{i}-\bar{z}\right)^{2}}_{E S S}$
      - residual sum of squares (RSS) is how much the real data still differs from your fitted data---the "unexplained variance"
   4. Initialization
      1. Zero or tiny final layer, to maximize entropy


5. Q - learning
   - Optimize memory usage carefully: you'll need it for replay buffer Learning rate schedules
   - Exploration schedules
   - Be patient. DQN converges slowly
     - On Atari, often $10-40 \mathrm{M}$ frames to get policy much better than random


6. Tips
   - Techniques from supervised learning don’t necessarily work in RL: batch norm, dropout, big networks
   - 超参数搜索：randomly 搜索所有超参数，然后观察各组实验结果，再用统计学指标，衡量最有影响的超参数。