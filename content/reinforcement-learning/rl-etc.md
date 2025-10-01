+++
date = '2025-10-01T16:20:45+09:00'
title = '11. RL Appendix(temp)'
draft= true
weight = 14
tags = ["Definition", "appendix", "Reinforcement Learning", "RL"]
categories = ["Reinforcement Learning"]
+++



# **appendix**

{{% hint warning %}}
// NOTE: 이 페이지는 임시로 작성되었습니다.
{{% /hint %}}


## **Model-Agnostic Meta-Learning(MAML)**
모든 task에 잘 작동하는 하나의 모델을 학습하는 것"이 아니라, <br>
조금만 fine-tuning 하면 각 task에 잘 작동할 수 있는 초기 모델"을 학습하는 것. 

## **PER**
위에 언급 [ 생략 ]


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


## **Proximal Policy Optimization (PPO)**
Actor-Critic 구조에서
Clipped Objective를 도입해서, policy가 너무 크게 바뀌지 않도록 제약하는것.

아래는 PPO surrogate objective 함수(목적함수). 이를 gradient ascent(최대화) 하는 파라미터를 현재 policy에 업데이트하면 policy는 옵티멀을 향해간다.
{{< katex display = true >}}
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
{{< /katex >}}




## **Deep RL**

RL과 딥러닝의 결합 <br>
딥러닝이 사용하는 위치
- **Policy Network** : 현재 상태에서 행동 분포를 출력
- **Value Network** : 상태나 상태-행동의 가치를 출력
- **Q-network** : Q(s, a)를 직접 추정
- **Model Network** : 환경 dynamics (transition, reward)를 예측 (model-based RL에서만 사용)

## **Learned Query Optimizer**

### **a. Balsa**
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



### **b. LOGGER**

- e-beam search 소개 [Exploration and exploitation]
- loss function reward weighting을 통해서 poor operator에 의한 fluctuation 방지
- log transformation 을 통해서 reward의 범위를 압축 (재앙적 plan의 영향도를 감쇄)
- ROSS Restricted Operator Search Space. (최적을 찾지 않고 최악이 안골라지게 해서 효율적)
- value nertwork (GNN + LSTM)

### **c. RELOAD**
Balsa + MAML + PER



## 참조
https://davidstarsilver.wordpress.com/teaching/

https://wnthqmffhrm.tistory.com/10

https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-5-model-free-control-.pdf