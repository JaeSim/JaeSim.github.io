+++
date = '2025-09-12T11:39:17+09:00'
title = '5. Model Free Control'
weight = 8
tags = ["Definition", "Model Free", "Monte Carlo", "Temporal Difference", "Reinforcement Learning", "RL"]
categories = ["Reinforcement Learning"]
+++

# **5. model-free control**
 - 이전장의 model-free prediction은 결국 예측하는 것이고. 이것은 실제로 행동하는것.
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
### **Monte-Carlo iteration**
- 이전장들중 policy interation 을 베이스로 한다.
- DP 대신에 Monte-Carlo evaluation을 넣는다.

Monte-Carlo 방법을 통해서 Policy Evaluation은 가능.(=Monte-Carlo Evaluation)<br>
greedy 하게 policy improvement는 action-value 펑션에 대해서만 가능하다.
state-value function을 하려면, 모델에 대해 알아야만 가능하다.(MDP를 알아야해)<br>

{{< katex display =true >}}
\pi'(s) = \arg\max_{a \in \mathcal{A}} \mathcal{R}_s^a + \sum_{s'} \mathcal{P}_{ss'}^a V(s') 
{{< /katex >}}
위와 대비되게 action-value function(Q 펑션) 은 model-free해서 알수 있다.


{{< katex display =true >}}
\pi'(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)
{{< /katex >}}

이렇게 알게된 policy를 아래 {{< katex display=false >}}\varepsilon{{< /katex >}}-greedy  방식으로 improvement 한다.

### **ε-greedy**
 - **강화학습에서 결국 옵티멀한 폴리시, 옵티멀한 value function을 가지려면, 모든 state에 대해서 탐험이 가능해야한다 <br>
 단순 greedy만으로는 이 조건을 만족 할 수 없다.**

 - 수식적으로도 입실론 그리디를 쓰면 다음 qvalue가 더 좋아진다는 것이 증명이 가능하다.
**{{< katex display=false >}}\varepsilon{{< /katex >}}-greedy Algo**

항상 최고의 행동만 고르면 탐험이 부족하고, 항상 무작위로 고르면 성능이 낮다. → 둘 사이를 적절히 섞자!

예시 )<br>
ε=0.1 (10%) <br>
90% 확률로 현재 최적의 행동, 10% 확률로 랜덤 행동

{{< katex display =true >}}
\pi(a \mid s) =
\begin{cases}
\frac{\epsilon}{m} + 1 - \epsilon & \text{if } a^* = \arg\max_{a \in \mathcal{A}} Q(s, a) \\
\frac{\epsilon}{m} & \text{otherwise}
\end{cases}
{{< /katex >}}

이것은 수학적으로 policy가 점차 좋아지는것을 나타내고 수렴한다는 계산 증명이 가능하다

 - 보통 decay되는 입실론 그리드를 많이 사용한다. (처음에는 탐험을 조금하지만 점점 안하도록)

### **Monte-Carlo Control**
 - Monde-Carlo Policy iteration 은 이전 섹션에서 설명한것
 - MC policy interation은 여러 episode 에 대해서 돌리고 업데이트를 하는데, <br>
 생각해보니 하나의 episode만 가지고도 q-value를 업데이트 할만하기 충분하다는 직관

<img src="/images/rl-mc-control.png" alt="rl-mc-control" style="width:80%;" />

Monde-Carlo Control 은 하나씩의 episode 가 끝난후에 policy를 업데이트하는것 (episode 단위로 policy improvement)<br>
이렇게 해도 되는 이유는(수렴하는이유는) `Greedy in the Limit with Infinite Exploration` 성질을 만족하기 때문이다..<br>
(policy를 업데이트하기 위한 충분한 정보를 이미 가지고 있다 라고 볼수도 있다.) <br>

#### **Greedy in the Limit with Infinite Exploration (GLIE)**
 - 지속적으로 시간이 지나면 결국 모든 state에 대해서 방문하니 방문 횟수가 무한대로 간다는 첫번째 성질
{{< katex display =true >}}
\lim_{k \to \infty} N_k(s, a) = \infty
{{< /katex >}}
 - 그리디하게 탐험하니 결국 policy가 수렴한다는 성질
 {{< katex display =true >}}
\lim_{k \to \infty} \pi_k(a \mid s) = \mathbf{1}\left(a = \arg\max_{a' \in \mathcal{A}} Q_k(s, a')\right)
{{< /katex >}}



### **TD Control : Sarsa**

 - MC보다 적은 분산을 가지고 있고, 한 episode가 끝나기 전에 스탭마다 폴리시를 업데이트가 가능

MC Control은 episode가 끝날때마다 정책을 개선하는데
Salsa는 매 step마다 정책을 개선.
<div style="text-align:center;">
  <img src="/images/rl-salsa.png" alt="rl-salsa" style="width:10%;"  />
</div>
{{< katex display =true >}}
Q(S, A) \leftarrow Q(S, A) + \alpha \left( R + \gamma Q(S', A') - Q(S, A) \right)
{{< /katex >}}

 - TD람다와 같이 Sarsa는 람다가 적용해서 Sarsa({{< katex display = false >}}\lambda{{< /katex >}}) 도 있다
 - 이것을 forward view로 보면 빠르게 업데이트하는(?) step마다 업데이트하는 Sarsa의 장점을 보이기때문에 계산 및 활용이 어려우나, <br>
 backward view 의 eligibility trace의 경우 이전 step에 대해서 responsibility assignment 하기 쉬움
 - Salsa람다는 sarsa의 기본과 eligibility trace까지 같이 업데이트 하는 것으로 이해

## **off-policy**
- 다른 agent나 휴먼으로부터 학습하할 수 있는 장점이 있음
- 경험을 재사용이 가능함
- 아직 가지못한 state에 대한 경험 같은것들을 배울수 있음 (더 탐험적 경험을 할수 있다고 이해)
- 하나의 policy에 대해서도 여러가지 경험들이 있으니 이것들을 학습에 이용할 수 있음

**behaviour policy µ를 통해서 수집하고, target policy π 를 학습하는것** <br>
이것은 다른 분포로부터 학습하는 성질을 이용
{{< katex display = true >}}
\mathbb{E}_{X \sim P}[f(X)]
= \sum P(X) f(X)
= \sum Q(X) \frac{P(X)}{Q(X)} f(X)
= \mathbb{E}_{X \sim Q} \left[ \frac{P(X)}{Q(X)} f(X) \right]
{{< /katex >}}


### **off-policy MC**
- Monte-Carlo 의 경우,
다음과 같이 value function 계산에 주입된다. 
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

### **off-policy TD**
- TD 의 경우 다음과 같이 계산된다
{{< katex display = true >}}
V(S_t) \leftarrow V(S_t) +
\alpha \left(
  \frac{\pi(A_t \mid S_t)}{\mu(A_t \mid S_t)}
  \left( R_{t+1} + \gamma V(S_{t+1}) \right)
  - V(S_t)
\right)
{{< /katex >}}

 - MC보다 분산이 적은 off-policy TD를 더 씀. + 데이터 효율성 극대화

### **Q-learning : Sarsa-max**
 - TD(0) = Sarsa 의 특별한 모양이 Q-learning
 - Sara-max 라고도 부름
 - importance sampling이 필요하지 않음

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



- target policy가 greedy 라면 아래와 같고
{{< katex display = true >}}
\pi(S_{t+1}) = \arg\max_{a'} Q(S_{t+1}, a')
{{< /katex >}}

- behavior policy가 입실론-greedy라면 다음과 같다.
{{< katex display = true >}}
R_{t+1} + \max_{a'} \gamma Q(S_{t+1}, a')
{{< /katex >}}

- 아래와 같이 표현이 되고,( max가 되는 action을 고르는 ), 동일하게 수렴한다는 보장 할 수 있다.
<div style="text-align:center;">
  <img src="/images/rl-q-learning.png" alt="rl-q-learning" style="width:20%;"  />
</div>
{{< katex display = true >}}
Q(S, A) \leftarrow Q(S, A) + \alpha \left( R + \gamma \max_{a'} Q(S', a') - Q(S, A) \right)
{{< /katex >}}


<img src="/images/rl-control-wrapup.png" alt="rl-control-wrapup"  />