+++
date = '2025-06-12T10:57:55+09:00'
title = '4. Model Free Prediction'
subtitle =  '강화학습의 Model Free Prediction 에 대한 내용'
weight = 7
tags = ["Definition", "Model Free", "Monte Carlo", "Temporal Difference", "Reinforcement Learning", "RL"]
categories = ["Reinforcement Learning"]
+++

# **4. model-free prediction**

## **Introduce**
- 우리는 MDP를 풀고싶지만, 실제환경에 대한 MDP는 주어지지 않는다.<br>
이전장에서는 명확하게 아는 경우에 대해서 DP로 풀다는 것을 본것.
- policy에 대한 evaluation (estimate)만 한다. 다음장에서는 이것을 이용해서 control(find optimal)한다. <br>
3장 DP에 있는것처럼, Model-free prediction 하고, Model-free control 하는 순서로 진행된다.
- 두가지 Major model-free prediction (MC와 TD) 가 있다.
- ***episode*** : 에이전트가 시작 상태에서 행동을 시작해서, 어떤 종료 조건(End state)에 도달할 때까지의 전체 과정



## **Monte-Carlo Learning**
- 직관적으로 epdisode를 다 끝낸다음에 활용하는 것
- **Monte Carlo (MC)** : 한 에피소드가 끝날 때까지 기다린 후, 그 전체 리턴 값을 이용하여 value function을 업데이트<br>
MC는 **하나의 에피소드 전체**(시작 ~ 종료)를 관찰한 뒤, <br>
실제로 받은 reward들을 기반으로 학습합니다. <br>
환경 모델 없이, 경험만으로 value function이나 policy를 추정합니다. <br>
Monte-Carlo policy evaluation uses _empirical mean_ return
instead of expected return <br>
높은 variance와 zero bias를 가짐




### **Monte-Carlo Policy Evaluation**
직접 돌려보는것이기 때문에 Expectation이 아니라 empirical mean으로 계산 수 있다 

- 첫번째 방문만 기록해서 하는 방법, 여러번 방문에 카운팅하는 방법이 있다.

즉, 가장 손쉬운 방법으로 epside를 돌려보고 가능성들의 mean 값으로 처리 <br>
방문할때마다 횟수와 토탈 return을 늘리고, 이것의 평균을 통해 value function을 estimate한다.
lecture-4, 7 page

 - 돌려본 다음 이것을 업데이트 하려면 다음과 같다.
{{< katex display =true >}}
V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)} \left(G_t - V(S_t)\right)
{{< /katex >}}
  - 위 수식은 incremental mean과 같은 form <br>
{{< katex display =false >}}
\mu_k = \frac{1}{k} \sum_{j=1}^{k} x_j 
= \frac{1}{k} \left( x_k + \sum_{j=1}^{k-1} x_j \right)
= \frac{1}{k}\left(x_k + (k-1)\mu_{k-1}\right)
= \mu_{k-1} + \frac{1}{k}(x_k - \mu_{k-1})
{{< /katex >}}

 - 아래 수식은 forgetting을 하고 싶을 non-stationary 문제의 경우 쓸수 있는 수식이다.
{{< katex display =true >}}
V(S_t) \leftarrow V(S_t) + \alpha \bigl(G_t - V(S_t)\bigr)
{{< /katex >}}

Blackjack 예제처럼,
확률이나, 분포 그런것 전혀 없이 episode 로 부터 value function을 만들어냈다. (~500,000반복하면서)


<img src="/images/rl-mc-backup.png" alt="rl-mc-backup" style="width:80%;"/>

## **Temporal Difference**
- 직관적으로 epdisode를 수행하지 않고 (일부만 수행하고) 활용하는 것
- **Temporal Difference (TD)** : 에피소드가 끝나지 않아도, 다음 상태의 현재 추정 값을 사용해 바로 업데이트 <br>
incomplete episodes 를 **bootstraping**을 통해서 업데이트 <br>
아직 에피소드가 끝나지 않아서 나머지 예상되는 reward를 포함해서 value function을 업데이트함 <br>
따라서 bias가 있음 + 낮은 variance를 가짐<br>
이것은 Markov property를 활용한다.
{{% hint info %}}
강화학습에서 **bootstraping**은 아직 끝나지 않은 미래값 (추정치)를 통해서 현재까지의 보상을 통해서 업데이트 하는것을 의미
{{% /hint %}}
- 이러한 TD 러닝도 결국에는 옵티멀한 value function을 찾는다. contraction mapping 때문이라는데,,

{{< katex display =true >}}
V(S_t) \leftarrow V(S_t) + \alpha \bigl(G_t - V(S_t)\bigr)
{{< /katex >}}

- 위 수식을 TD는 episode를 돌리지 않고 해야하니 아래와 같이 변환할 수 있다.

{{< katex display =true >}}
V(S_t) \leftarrow V(S_t) + \alpha \left( R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right)
{{< /katex >}}

<img src="/images/rl-td-backup.png" alt="rl-td-backup" style="width:80%;" />

## **MC vs TD**
 - MC 는 전체 epsido가 끝난뒤 계산
 - TD는 현재 step으로부터 계산
 - TD는 에피소드를 다 기다리지 않고 할수 있으므로 샘플 효용성이 높음. 단 바이어스가 있음 (한 스탭만 하고 예측하기 때문에, 낮은 분산)
<img src="/images/rl-mc-td-compare.png" alt="rl-mc-td-compare"  />

- 참고용 DP의 backup

<img src="/images/rl-dp-backup.png" alt="rl-dp-backup" style="width:80%;"/>

- MC, TD, DP 등의 관계
<img src="/images/rl-backup-category.png" alt="rl-backup-category" style="width:80%;" />


-  Monte-Carlo Reinforcement Learning 은 model-free 이다.
왜냐하면 MDP Transition 에 대한 (reward에 대한) 지식이 없기 때문.
- Temporal-Difference Learning 또한 model-free



## **TD-람다**
- 직관적으로 MC와 TD의 중간단계
- **TD({{< katex display = false >}}\lambda{{< /katex >}})** : 여러 step + 가중합으로 업데이트 ?  MC와 TD의 중간<br>
MC <-> TD는 전체 에피소드를 보느냐, 일부분만 보느냐의 차이.<br>
TD의 step을 0~n (n이 되면 MC와 같음) 사이를 {{< katex display = false >}}\lambda{{< /katex >}}로 가중치를 구해 사용하는것
- n-step TD 에서의 n은 몇개의 step까지 하고 그다음을 예측할지를 나타내고, 람다는 {{< katex display = false >}}\lambda{{< /katex >}} 는 그 n-step들의 가중합을 하기 위한 적당한 값. (멀리있는것을 더 작게 영향이 가도록)

<img src="/images/rl-td-lambda.png" alt="rl-lambda" style="width:50%;" />

- 이것을 수식으로 나타내면 다음과 같다.
{{< katex display = true >}}
G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{\,n-1} G_t^{(n)}
{{< /katex >}}
- 그리고 이것을 TD Learning에 수식에 넣으면 다음과 같다.
{{< katex display = true >}}
V(S_t) \leftarrow V(S_t) + \alpha \left( G_t^\lambda - V(S_t) \right)
{{< /katex >}}

- TD 람다는 미래의 것을 반영한다는 직관이 있지만, 반대로 backward view에서는 현재 발생한 현상에 대해서, 과거의 영향이 있다는 직관이 있다.
- 따라서 현재것을 학습할때, 과거의 비율만큼 나누어 업데이트(프로파게이션)가 될 수 있다.
- 왼쪽이 backward, 오른쪽이 forward 인데 둘다 수학적으로 동일하게 된다는 것이 증명 되어 있다. <br>
 forward는 미래의 것을 하는것이므로, 왼쪽 backward가 더 계산이 용이해서 많이 사용된다.
{{< katex display = true >}}
\sum_{t=1}^T \alpha \delta_t E_t(s) = \sum_{t=1}^T \alpha \left( G_t^\lambda - V(S_t) \right) \mathbf{1}(S_t = s)
{{< /katex >}}

- TD 람다에서 람다가 0 이라면 바로 앞의 것만 사용해서 동작하는 것. 1 이 라면 MC