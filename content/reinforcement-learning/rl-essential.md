+++
date = '2025-05-22T11:13:00+09:00'
title = '1. Reinforcement Learning Essential'
subtitle =  '강화학습에 대한 기초 내용'
weight = 4
tags = ["Definition", "Essential", "Reinforcement Learning"]
categories = ["Reinforcement Learning"]
+++
# **1. 강화학습에 대한 기초 내용**

## **들어가기 앞서**

강화학습의 기초적인 내용을 학습한 뒤 정리한 것으로,
남들에게 보여주기 보다는 본인의 이해와 기억을 위해서 기술한 것입니다.

나만의 방식으로 이해한 것이기 때문에, 주요하다고 생각하는 부분이 다를수 있으며 생략되거나 놓친 부분이 많이 있습니다.


## **강화학습(RL: Reinforcement Learning) 이란?**
### **Definition**
> “Reinforcement learning is learning what to do—how to map **situations** to **actions**—so as to maximize a numerical **reward** signal.”  
> — *Richard S. Sutton and Andrew G. Barto*, _Reinforcement Learning: An Introduction_ (2nd ed), p.1



situations을 state로 표기하여

통상적으로 action, state와 reward 가 RL의 핵심 요소이다.

<!-- ![figure](/images/rl-action-state-reward.png) -->
<img src="/images/rl-action-state-reward.png" alt="rl-essential" style="width:80%;" />

요약하면, 
**주어진 상태(State)에서 보상(Reward)을 최대화 할 수 있는 행동(Action)을 학습하는 것**

이는 Reward Hypothesis를 기반으로한다.

>**Reward Hypothesis**<br>
>All goals can be described by the maximisation of expected cumulative reward

Rewards는 Scalar feedback signal이다
Agent는 미래에 기대되는 cumulative reward가 최대화 되는 방향으로 학습 한다


### **State and MDP**

- **Agent**는 학습하는 주체 (뇌, 로봇)
- **Environment**는 환경 (지구, 게임)    //우리는 환경이 어떻게 동작하는지 보통 모름
- **State** 는 다음 action을 결정하기 위한 정보이다    {{< katex display= true >}}S_t = f(H_t) {{< /katex >}}


**Agent가 주어진 상태(State)에서 보상(Reward)을 최대화 할 수 있는 행동(Action)을 학습하는 것**


**State** 에는 <br>
Environment State, Agent State 가 있고
각각 Environment, Agent관점에서의 수식적/내부적 표현이다.

**Environment State**는 Environment 관점에서 다음 step에서 Environment가 어떻게 변화할지를(어떤 State로 변화할지) 나타낸다.
Environment에서는 microsecond 에서도 수많은 정보가 오기 때문에 불필요한 정보들도 많다.
보통의 RL 문제에서 agent는 Environment state를 전부 관측할 수 없다.
 - 예) 로봇의 움직임을 학습하도록 설계했는데. 이 로봇은 지구의 중력이나, 마찰력 으로 물건이 어디로 움직이는지 정확하게 모른다.

**Agent State**는 다음 step 에서 Agent가 어떤 행동을 선택할지를 나타낸 수식/표현이다.


**Information State**는 과거 history부터 모든 유용한 정보를 포함한 수학적 정의를 가진 State이다. 주로 Markov State라 부른다. (Markov 속성을 만족한다)

- **Markov Properties** : <br>
이전의 모든 State 정보를 이용해서 다음 State를 선택하는것이, 현재 State만 보고 하는것과 같다
{{< katex display=true >}}
\mathbb{P}[S_{t+1} \mid S_t] = \mathbb{P}[S_{t+1} \mid S_1, \dots, S_t]
{{< /katex >}}

이를 이용하면, 미래(future) 는 과거에 무엇이 주어졌든지 독립적이다.
(=The future is independent of the past given the present)
달리말하면, 현재 {{< katex display=false >}}S_t{{< /katex >}} 만 저장해도 된다
{{< katex display=true >}}
H_{1:t} \rightarrow S_t \rightarrow H_{t+1:\infty}
{{< /katex >}}
현재의 State가 충분한 정보를 이미 담고 있다다.
{{% details title="Appendix" open=false %}}
- history 는 Observation과 actions, rewards의 연속이다
<div style="text-align: center;">
{{< katex display=true >}}
\quad \quad H_t = O_1, R_1, A_1, ..., A_{t−1}, O_t, R_t
{{< /katex >}}
</div>

- State 는 다음 action을 결정하기 위한 정보이다

<div style="text-align: center;">
{{< katex display=true >}}
\quad \quad S_t = f(H_t)
\\
{{< /katex >}}
</div>

- 이전 state는 
<div style="text-align: center;">
{{< katex display=true >}}
\mathbb{P}[S_{t+1} \mid S_t] = \mathbb{P}[S_{t+1} \mid S_1, \dots, S_t]
{{< /katex >}}
</div>

- The future is independent of the past given the present
- 모든 이전 State를 알지 않아도 직전 State만 보고 결정 할 수 있다

<div style="text-align: center;">
{{< katex display=true >}}
H_{1:t} \rightarrow S_t \rightarrow H_{t+1:\infty}
{{< /katex >}}
</div>
{{% /details %}}


**Fully Observable Environments** 는 Agent가 environment 어떻게 동작하는지 바로 관측이 가능함을 나타내고,
결과적으로 Environment State = Information State = Agent State 상태이다.
이를 **Markov Decision Process(MDP)** 라고 한다

**Partial Observable Environments** 는 좀더 현실적인 환경. 로봇이 카메라를 통해서 화면을 보지만 현재 자기의 위치를 모르는 것처럼.
즉, Agent State {{< katex display=false >}} \ne {{< /katex >}} Environment State 이다.
**Partially Observable Markov Decision Process(POMDP)** 로 수식이 표현된다.

Agent는 자기의 State에 대한 representation을 가져야만하고, 이는
- 이전 History를 이용해서 사용하는 방법이 있고,
{{< katex display=true >}}S_t^a = H_t{{< /katex >}}
- Probability 로 나타내는 방법이 있고,
{{< katex display=true >}}S_t^a = \left( \mathbb{P}[S_t^e = s_1], \dots, \mathbb{P}[S_t^e = s_n] \right){{< /katex >}}
- 순환신경망(Recurrent neural network) 으로 나타내는 방법도 있다.
{{< katex display=true >}}S_t^a = \sigma(S_{t-1}^a W_s + O_t W_o){{< /katex >}}


### **Policy, Value Function, Model**

<img src="/images/rl-action-state-reward-with-component.png" alt="rl-essential" style="width:80%;" />


RL은 agent는 아래 components를 _**한개 이상**_ 포함한다.
> **Policy**는 **_Agent가 어떻게 Action을 선택하는지(=behaviour function)_** 이다.

State로부터 function {{< katex display=false >}}\pi{{< /katex >}}(policy)를 이용해서를 통해 action을 결정한다.
- Deterministic policy: {{< katex display=false >}} a = \pi(s){{< /katex >}} <br>
 state를 넣으면 다음 취할 액션이 튀어나온다

probability로 policy를 표현하고자 한다면 다음과 같다.
- Stochastic policy: {{< katex display=false >}} \pi(a \mid s) = \mathbb{P}[A_t = a \mid S_t = s]{{< /katex >}} <br>
 현재 상태 s 에 있을 때, 액션 a 를 선택할 확률

> **Value Function**은 **_State나 Action이 얼마나 좋은지(기대되는 미래의 reward가 얼마일지 예측)_** 을 나타낸다.

State one과 State two,  action one과 action two를 선택할때 최종 reward가 더 좋은쪽으로 선택한다.

아래와 같이 표현되는데 {{< katex display=false >}} R_{t+1}, R_{t+2}, R_{t+3} {{< /katex >}} 를 더하는 것과 같이 다음(미래)의 reward의 합의 **기대값**(어떤 폴리시 {{< katex display = false >}}\pi{{< /katex >}}를 따르는 가정하에)
{{< katex display=true >}}
v_{\pi}(s) = \mathbb{E}_{\pi} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \mid S_t = s \right]
{{< /katex >}}
gamma({{< katex display=false >}} \gamma{{< /katex >}}) 는 다음 스탭에 대한 discounting factor (지금은 미래의 값의 영향도 정도로 이해하고 넘어가자)

> **Model**은 **_Agent관점에서 Environment가 어떻게 동작할지 생각하는 것_** 을 나타낸다.

transitions model, rewards model 전통적으로 두가지로 나뉜다

transitions 모델은 directly 다음 state를 예측한다.
{{< katex display=true >}}\mathcal{P}_{ss'}^a = \mathbb{P}[S_{t+1} = s' \mid S_t = s, A_t = a]{{< /katex >}}
Rewards 모델은 reward를 예측한다.
{{< katex display=true >}}\mathcal{R}_s^a = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]{{< /katex >}}

### **RL agent의 분류**

<img src="/images/rl-agent-taxonomy.png" alt="rl-agent-taxonomy" style="width:80%;" />

어떤 key component를 가지고 학습하고 있는지에 따라서 RL을 분류한다
- value-based RL은 value function을 optimal이 되도록 한다. (묵시적으로 policy 를 가지고 있다.)
- policy-based RL은 policy를 업데이트한다.
- actor-critic은 value function과 policy를 가지고 있다.

Model base로 구분하는 방법이 있다.
- model free는 model이 없지만(=환경에 대한 representation이 없지만) value function and/or policy로 구성된 RL 
- model based는 model이 존재하고, value function and/or policy 로 구성된 RL


### **Sequential decision making의 두가지 방식**
- **Reinforcement Learning**은 환경(Environment)을 모르고 상호작용하면서 reward가 최대가 되도록 학습 하는 것
- **Planning**은 환경을 알고(우리가 환경에 해당하는 rule/model을 주고) agent가 계산하는 것


### **Exploration and Exploitation**
Exploration 와 Exploitation 는 trade-off
- Exploration 은 환경에 대한 정보를 더 찾는것
- Exploitation 은 알고 있는 정보를 활용해서 reward를 최대화 하는것

예) 새로운 레스토랑 찾기 vs 가장 좋아하는 레스토랑 재방문

### **Prediction and Control**
RL에서는 prediction problem과 control problem 이 있다.

- **prediction** 은 현재 policy 를 따르면 앞으로 얼마나 미래에 좋을지 평가하는것
- **contorl** 은 bset policy(=optimal policy)를 찾는것

---

## **머신러닝(ML)과 딥러닝(DL)과의 관계**

머신러닝(Machine Learning)은 인공지능(AI)의 개념으로써, 학습을 통해 예측(또는 분류)를 하는 것

딥러닝은 머신러닝의 하위 개념으로써, 인공신경망(Neural Network)를 이용해서 학습하는 것

강화 학습은 머신러닝의 한갈래로써, 보상을 기반으로 스스로 행동 학습하는 것
```
AI
├── Machine Learning
│   ├── Supervised / Unsupervised
│   ├── Reinforcement Learning
│   └── Deep Learning
│       └── Deep Reinforcement Learning (e.g., DQN, PPO)
```


## **참고**

David Silver 의 RL 강좌
https://davidstarsilver.wordpress.com/teaching

한글 유튜브 + 블로그
https://smj990203.tistory.com/2