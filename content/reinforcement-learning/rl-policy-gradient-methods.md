+++
date = '2025-09-23T13:22:02+09:00'
title = '7. Polciy Gradient Methods (Not completed)'
weight = 10
tags = ["Definition", "Model Free", "Value Function", "policy gradient", "Reinforcement Learning", "RL"]
categories = ["Reinforcement Learning"]
+++

# **7. Policy Gradient Method**

{{% hint warning %}}
// NOTE: 이 페이지는 임시로 작성되었습니다.
{{% /hint %}}


이전섹션에서는 action-value function(state value function, action-value function)을 근사해서 옵티멀한 value function을 찾아 갔다. 이때 policy는 {{< katex display =false >}}\epsilon{{< /katex >}}-greedy. 


policy gradient는 policy parameter를 직접 업데이트 한다. **policy-based !** <br>
policy를 direct 로 업데이트 하는것 <br>
이는 large complicated environment에 좀더 적합. 각 state를 명확하게 구분하기 어려울때. <br>
 (Policy는 value 대비 더 compact하게 저장한다)
 
- policy는 probability 이며 현재 state에서 action을 선택할 확률 이고 action들이니 확률 분포이다.
{{< katex display =true >}}
\pi_\theta(s,a) = \mathbb{P}[a \mid s, \theta]
{{< /katex >}}
<img src="/images/rl-value-policy-diagrampng.png" alt="policy, value and action cricit" style="width:80%;"/>


 - Value-Based and Policy-based RL은 다음과 같은 특성을 가지고 있다.

| Method            | Value Function 여부      | Policy 여부         | 
|----------------|-------------------------------|----------------------------|
| **Value-Based** | Learnt Value Function   | Implicit Policy<br> (e.g. {{< katex display =false >}}\epsilon{{< /katex >}}-greedy)      |
| **Policy-Based**| No Value Function    | Learnt Policy        |
| **Actor-Critic**| Learnt Value Function   | Learnt Policy        |




- Softmax Policy : 이산행동일때, softmax over logits으로 정책을 정하고, policy에 대한 gradient 값 계산<br>
이를 통해서 현재 iteration 에 대한 policy 파라미터를 업데이트한다.
- Gaussian Policy : 연속행동일때,  가우시안 분포를 정책으로 정하고, policy에 대한 gradient 값 계산<br>
이를 통해서 현재 iteration 에 대한 policy 파라미터를 업데이트한다.

{{% hint warning %}}
policy-based 에서도 replay buffer를 쓸수 있는가?

1) Policy-based에서 Replay Buffer의 어려움
문제: Policy가 계속 변하기 때문에
​
2) PPO, TRPO [	❌ 또는 제한적 사용 ]	최근의 데이터만 사용 (very short buffer)

3) SAC (Soft Actor-Critic) [ 확률적 policy 사용 ] (policy gradient 기반) <br>
하지만 전체 구조는 off-policy 로써 replay buffer 적극 사용

by GPT

**추가 research 요망**
{{% /hint %}}


## **Actor-Critic**
policy-based + value-based

Actor는 행동을 생성하고 <br>
Critic은 그 행동의 "좋음"을 평가해서 Advantage를 추정 <br>
이를 바탕으로 Actor의 policy gradient를 계산 <br>

lacture-7 24page

간단한 actor-critic 구조는 actuion-value를 critic 하는것이다.
Critic은 TD(0)을 통해서 Q-Function을 학습하고
Actor는 위 학습된 Q-function을 기반으로 policy gradient를 수행하는것이다.

### **Proximal Policy Optimization (PPO)**
Actor-Critic 구조에서
Clipped Objective를 도입해서, policy가 너무 크게 바뀌지 않도록 제약하는것.

아래는 PPO surrogate objective 함수(목적함수). 이를 gradient ascent(최대화) 하는 파라미터를 현재 policy에 업데이트하면 policy는 옵티멀을 향해간다.
{{< katex display = true >}}
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
{{< /katex >}}

# **8. Integration Learning and Planning**

## **Model-based RL**
**[생략]**

## **Simulation-based Search**
"전체 상태 공간을 직접 학습하기는 너무 비싸다. 대신, 유망한 부분만 집중적으로 시뮬레이션하면서 거기서 경험한 정보로 Q값을 점점 더 정확하게 만든다."

//note: balsa simulation learning 과는 개념이 조금 다름

**[미완]**

# **9. Exploration and Exploitation**
- **Exploitation** : Make the best decision given current information
- **Exploration** : Gather more information


너무 탐험만 하면: 성능이 낮은 행동도 계속 시도 → 학습은 느리고, 보상은 낮음 <br>
너무 이용만 하면: 더 나은 행동을 아예 시도하지 않음 → **지역 최적해(local optimum)**에 갇힘 <br>
→ 따라서, 단기 보상 vs 장기 학습 사이의 균형을 잡는 것이 중요합니다. <br>

이 섹션에서는 여러 방법들을 제시하고 있습니다
[미완]


# **10. Case Study: RL in Classic Games**
[미완]

---
# **999. Deep RL**

RL과 딥러닝의 결합 <br>
딥러닝이 사용하는 위치
- **Policy Network** : 현재 상태에서 행동 분포를 출력
- **Value Network** : 상태나 상태-행동의 가치를 출력
- **Q-network** : Q(s, a)를 직접 추정
- **Model Network** : 환경 dynamics (transition, reward)를 예측 (model-based RL에서만 사용)

# **a. Balsa**
Balsa는 쿼리 플랜을 순차적으로 구성하는 문제를 **Markov Decision Process (MDP)** 으로 보고,
이를 강화학습으로 해결

- State s = 현재까지 만들어진 partial query plan
- Action a = 다음에 어떤 테이블을 조인할지 결정
- Reward r = 쿼리 플랜의 실행 비용 또는 latency
- Environment = DB 쿼리 시뮬레이터 or Costmodel


추가적으로 
- simulation phase(step) 을 가져서 재앙적 plan을 탐험하지 않게하고,
- Timeout을 둬서 Safe Execution 시간을 보장했다. (재앙적 plan이 선택되더라도 timeout으로 하한보장)
- value network를 simple tree convolution networks 로 구성

```python
# 여기에서 모델은 강화학습의 environment의 모델이 아니라, value function을 근사할(계산할) treeconv 모델을 의미함
def MakeModel(p, exp, dataset):
    dev = GetDevice()
    num_label_bins = int(
        dataset.costs.max().item()) + 2  # +1 for 0, +1 for ceil(max cost).
    query_feat_size = len(exp.query_featurizer(exp.nodes[0]))
    batch = exp.featurizer(exp.nodes[0])
    assert batch.ndim == 1
    plan_feat_size = batch.shape[0]

    if p.tree_conv:
        labels = num_label_bins if p.cross_entropy else 1
        return TreeConvolution(feature_size=query_feat_size,
                               plan_size=plan_feat_size,
                               label_size=labels,
                               version=p.tree_conv_version).to(dev)
```



# **b. LOGGER**

- e-beam search 소개 [Exploration and exploitation]
- loss function reward weighting을 통해서 poor operator에 의한 fluctuation 방지
- log transformation 을 통해서 reward의 범위를 압축 (재앙적 plan의 영향도를 감쇄)
- ROSS Restricted Operator Search Space. (최적을 찾지 않고 최악이 안골라지게 해서 효율적)
- value nertwork (GNN + LSTM)

# **c. RELOAD**
Balsa + MAML + PER

## **Model-Agnostic Meta-Learning(MAML)**
모든 task에 잘 작동하는 하나의 모델을 학습하는 것"이 아니라, <br>
조금만 fine-tuning 하면 각 task에 잘 작동할 수 있는 초기 모델"을 학습하는 것. 

## **PER**
위에 언급 [ 생략 ]

## 참조
https://davidstarsilver.wordpress.com/teaching/

https://wnthqmffhrm.tistory.com/10

https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-5-model-free-control-.pdf