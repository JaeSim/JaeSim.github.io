+++
date = '2025-06-12T10:57:55+09:00'
title = 'temp. 4. Model Free Prediction'
subtitle =  '강화학습의 Model Free Prediction 에 대한 내용'
weight = 7
tags = ["Definition", "Model Free", "Monte Carlo", "Temporal Difference", "Reinforcement Learning"]
categories = ["Reinforcement Learning"]
+++

# **4. model-free prediction**



{{% hint warning %}}
// NOTE: 이 페이지는 임시로 작성되었습니다.
{{% /hint %}}


3장 DP에 있는것처럼, Model-free prediction 하고, Model-free control 하는 순서로 진행된다.

episode : 에이전트가 시작 상태에서 행동을 시작해서, 어떤 종료 조건(End state)에 도달할 때까지의 전체 과정

MC와 TD는 major model-free algo.

- **Monte Carlo (MC)** : 한 에피소드가 끝날 때까지 기다린 후, 그 전체 리턴 값을 이용하여 value function을 업데이트<br>
MC는 **하나의 에피소드 전체**(시작 ~ 종료)를 관찰한 뒤, <br>
실제로 받은 reward들을 기반으로 학습합니다. <br>
환경 모델 없이, 경험만으로 value function이나 policy를 추정합니다. <br>
Monte-Carlo policy evaluation uses _empirical mean_ return
instead of expected return <br>
높은 variance와 zero bias를 가짐


즉, 가장 손쉬운 방법으로 epside를 돌려보고 가능서들의 mean 값으로 처리 <br>
방문할때마다 횟수와 토탈 return을 늘리고, 이것의 평균을 통해 value function을 estimate한다.
lecture-4, 7 page
{{< katex display =true >}}
V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)} \left(G_t - V(S_t)\right)
{{< /katex >}}

Blackjack 예제처럼,
확률이나, 분포 그런것 전혀 없이 episode 로 부터 value function을 만들어냈다. (~500,000반복하면서)

- **Temporal Difference (TD)** : 에피소드가 끝나지 않아도, 다음 상태의 현재 추정 값을 사용해 바로 업데이트 <br>
incomplete episodes 를 bootstraping을 통해서 업데이트 <br>
아직 에피소드가 끝나지 않아서 나머지 예상되는 reward를 포함해서 value function을 업데이트함 <br>
따라서 bias가 있음 + 낮은 variance를 가짐<br>
이것은 Markov property를 활용한다.

{{< katex display =true >}}
V(S_t) \leftarrow V(S_t) + \alpha \left( R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right)
{{< /katex >}}


- **TD({{< katex display = false >}}\lambda{{< /katex >}})** : 여러 step + 가중합으로 업데이트 ?  MC와 TD의 중간<br>
MC <-> TD는 전체 에피소드를 보느냐, 일부분만 보느냐의 차이.<br>
TD의 step을 0~n (n이 되면 MC와 같음) 사이를 {{< katex display = false >}}\lambda{{< /katex >}}로 가중치를 구해 사용하는것<br>
Monte-Carlo Reinforcement Learning 은 model-free 이다.
왜냐하면 MDP Transition 에 대한 (reward에 대한) 지식이 없기 때문.

Temporal-Difference Learning 은 model-free


<img src="/images/rl-backup-category.png" alt="rl-essential" style="width:80%;" />


# **5. model-free control**
## **on-policy/off-policy intro**

π = Target Policy
µ = Behavior Policy

이두개가 같으면 on-policy 다르면 off-policy.

- Off-policy learning : “Look over someone’s shoulder” <br>
Learn about policy π from experience sampled from µ <br>
**re-use** experience generated from old policy <br>
**Q-Learning** : {{< katex display=false >}}\varepsilon{{< /katex >}}-greedy 방식으로 탐험하지만 학습에는 반영 안할수 있음(최적의 행동만 업데이트)

- On-policy learning : “Learn on the job” <br>
Learn about policy π from experience sampled from π<br>
**Salsa** :on-policy Q-learning. 현재 행동을 그대로 따라가며 학습

## **on-policy**
### **Monte-carlo iteration**
Monte-Carlo 방법을 통해서 Policy Evaluation은 가능.(=Monte-Carlo Evaluation)<br>
greedy 하게 policy improvement는 action-value 펑션에 대해서만 가능하다.
state-value function을 하려면, 모델에 대해 알아야만 가능하다.<br>

{{< katex display =true >}}
\pi'(s) = \arg\max_{a \in \mathcal{A}} \left( \mathcal{R}_s^a + \sum_{s'} \mathcal{P}_{ss'}^a V(s') \right)
{{< /katex >}}
위와 대비되게 action-value function(Q 펑션) 은 model-free해서 알수 있다.


{{< katex display =true >}}
\pi'(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)
{{< /katex >}}

이렇게 알게된 policy를 아래 {{< katex display=false >}}\varepsilon{{< /katex >}}-greedy  방식으로 improvement 한다.

### **ε-greedy**
**{{< katex display=false >}}\varepsilon{{< /katex >}}-greedy Algo**

항상 최고의 행동만 고르면 탐험이 부족하고, 항상 무작위로 고르면 성능이 낮다. → 둘 사이를 적절히 섞자!

예시 )<br>
ε=0.1 (10%) <br>
90% 확률로 현재 최적의 행동 <br>
10% 확률로 랜덤 행동 <br>

이것은 수학적으로 policy가 점차 좋아지는것을 나타내고 수렴한다는 계산 증명이 가능하다

### **Monte-carlo Control**
Monde-Carlo Policy iteration 은 이전 섹션에서 설명한것

Monde-Carlo Control 은 하나씩의 episode 가 끝난후에 policy를 업데이트하는것 (episode 단위로 policy improvement)<br>
이렇게 해도 되는 이유는(수렴하는이유는) `Greedy in the Limit with Infinite Exploration` 성질을 만족하기 때문이다..<br>
(policy를 업데이트하기 위한 충분한 정보를 이미 가지고 있다 라고 볼수도 있다.) <br>
[화살표 그림 추가예정]

GLIE 성질에 대한 설명은 추후 업데이트.

### **Sarsa**

MC Control은 episode가 끝날때마다 정책을 개선하는데
Salsa는 매 step마다 정책을 개선.


## **off-policy**
**behaviour policy µ를 통해서 수집하고, target policy π 를 학습하는것** <br>
이것은 다른 분포로부터 학습하는 성질을 이용
{{< katex display = true >}}
\mathbb{E}_{X \sim P}[f(X)]
= \sum P(X) f(X)
= \sum Q(X) \frac{P(X)}{Q(X)} f(X)
= \mathbb{E}_{X \sim Q} \left[ \frac{P(X)}{Q(X)} f(X) \right]
{{< /katex >}}

이것은 다음과 같이 value function 계산에 주입된다.
{{< katex display = true >}}
G_t^{\pi / \mu} =
\frac{\pi(A_t \mid S_t)}{\mu(A_t \mid S_t)}
\frac{\pi(A_{t+1} \mid S_{t+1})}{\mu(A_{t+1} \mid S_{t+1})}
\cdots
\frac{\pi(A_T \mid S_T)}{\mu(A_T \mid S_T)}
G_t
{{< /katex >}}
{{< katex display = true >}}
V(S_t) \leftarrow V(S_t) + \alpha \left( G_t^{\pi / \mu} - V(S_t) \right)
{{< /katex >}}

### **Q-learning**
action-value Q(s,a) 의 off-policy learning <br>
다음 action을 선택할때 behaviour policy로부터 고르고, {{< katex display = false >}}A_{t+1} \sim \mu(\cdot \mid S_t){{< /katex >}} <br>
학습은 target policy 기반으로 진행. {{< katex display = false >}}A' \sim \pi(\cdot \mid S_t){{< /katex >}}<br>
**위와 같이 진행해도, Q-learing은 결국에는 <br>
옵티멀한 action-value (q) function에 수렴한다는 성질을 이용한것.** <br>

밑에는 improvement 하는 수식.
{{< katex display = true >}}
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( R_{t+1} + \gamma Q(S_{t+1}, A') - Q(S_t, A_t) \right)
{{< /katex >}}

이렇게 하면 복잡한 weight 계산식 을 하지 않아도 된다.



[마지막 랩업 표 추가 예정]
# **6. value function approximation**
large MDP를 풀수 없으니 (too many state and action) value function(state-value/action-value function)을 어떻게 근사하게 구하는가?
- Liner combinations of feature
- Neural network


여기서부터 Gradient Descent 가 나온다.

>1) 정책 π 고정 
>2) Q-function을 gradient descent로 근사 (policy evaluation) 
>3) 근사된 Q 기반으로 정책 개선 (policy improvement) 
>4) 다시 반복 (⇒ 점진적으로 최적 정책에 수렴)


value function을 근사해서 사용하므로 **value-based** 라고도 한다.

## **Action-value function Approximation**
아래와 같이 action value function은 approximate를 통해서 표현될수 있고.
델타 W를 작게하므로써 근사를 구할 수 있다.
{{< katex display = true >}}
J(\mathbf{w}) = \mathbb{E}_{\pi} \left[ \left( q_{\pi}(S, A) - \hat{q}(S, A, \mathbf{w}) \right)^2 \right]
{{< /katex >}}

{{< katex display = true >}}
- \frac{1}{2} \nabla_{\mathbf{w}} J(\mathbf{w}) 
= \left( q_{\pi}(S, A) - \hat{q}(S, A, \mathbf{w}) \right) \nabla_{\mathbf{w}} \hat{q}(S, A, \mathbf{w})
{{< /katex >}}
{{< katex display = true >}}
\Delta \mathbf{w} = \alpha \left( q_{\pi}(S, A) - \hat{q}(S, A, \mathbf{w}) \right) \nabla_{\mathbf{w}} \hat{q}(S, A, \mathbf{w})
{{< /katex >}}

## **Batch Method**
위의 gradient descent 할때 sampling을 효율적으로 하기 위한 여러가지 방법들

### **Experience Replay**
과거의 transition들을 버리지 않고 buffer에 저장해 두었다가,
학습할 때마다 랜덤하게 샘플링해서 사용

### **Prioritized Experience Replay (PER)**
모든 transition을 균등하게 샘플하지 않고,
TD-error가 큰 transition에 더 높은 확률을 부여하여 학습

직관: TD-error가 클수록 더 학습이 필요한 "중요한" 경험

# **7. policy gradient Method**

이전섹션에서는 action-value function(Q 펑션)을 근사해서 옵티멀한 폴리시를 찾아갔다면, <br>
policy gradient는 Q 없이 policy parameter를 직접 업데이트 한다. **policy-based**


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
- policy nertwork (GNN + LSTM)

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