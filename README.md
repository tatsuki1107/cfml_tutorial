# CounterFactual Machine Learning Tutorial

# 1. Bandit Algorithm

## On Policy
### Contextual Bandit Algorithm
<a href="https://github.com/tatsuki1107/cfml_tutorial/blob/master/bandit/policy/contextual_bandit.py">`contextual_bandit.py`</a>

### notebook
- <a href="https://github.com/tatsuki1107/cfml_tutorial/blob/master/bandit/continous_reward.ipynb">`continuous_reward.ipynb`</a>
- <a href="https://github.com/tatsuki1107/cfml_tutorial/blob/master/bandit/binary_reward.ipynb">`binary_reward.ipynb`</a>

### Explanatory article about the policy
- LinUCB:　<a href="https://qiita.com/tatsuki1107/items/02d51371f8db9eccfb30">`LinUCB方策のUCBスコア導出`</a>  
- LinTS: 　<a href="https://qiita.com/tatsuki1107/items/f720a01c4c851345ee32">`LinTS方策における事後分布(ガウス分布)の導出`</a>
- LogisticTS: <a href="https://qiita.com/tatsuki1107/items/b6bfc67be869ea6919e8">`LogisticTS方策におけるラプラス近似した事後分布(ガウス分布)の導出`</a>  

## Off Policy

### Estimator for Off Policy Evaluation
- <a href="https://github.com/tatsuki1107/cfml_tutorial/blob/master/bandit/ope_estimator/estimator.py">`estimator.py`</a>

### notebook
- <a href="https://github.com/tatsuki1107/cfml_tutorial/blob/master/bandit/ope.ipynb">`ope.ipynb`</a>

# 2. Basic OPE/OPL
### notebook
- <a href="https://github.com/tatsuki1107/cfml_tutorial/blob/master/basic/basic_ope_using_obp.ipynb">`basic_ope_using_obp.ipynb`</a>

### Explanatory article about the estimator
- DirectMethod(DM), InversePropensityScore(IPS), DoublyRobust(DR) Estimator: <a href="https://qiita.com/tatsuki1107/items/20fd7c6ccb7019f10766">`オフ方策評価におけるDM,IPS,DR推定量の期待値/分散の分析 (最後に「プラットフォーム全体で観測される報酬のオフ方策評価」と題してユニークなIPS推定量を導出しました)`</a>

# 3. Ranking


