+++
date = '2025-05-27T16:43:57+09:00'
title = '2. Markov Decision Process'
subtitle =  '강화학습의 MP, MRP, MDP에 대한 내용'
weight = 5
tags = ["Definition", "Markov", "Markov Process", "Markov Decision Process", "Reinforcement Learning"]
categories = ["Reinforcement Learning"]
+++

# **2. Markov Decision Process**

## **Markov Process(MP) 란?**
### **MP 속성**

MDP는 환경에 대해서 Reinforcement Learning이 이해가능하도록 수식화한다

<U>거의 모든 RL 관련 문제들은 MDP로 수식화 할 수 있다(Fully observable이나 Partially observation이나) </U>

Markov Property를 이용하는데, --이전 강의참조--

요약하면, 현재 state만으로 미래를 예측해도 된다는 속성이다.
(다르게 말하면, 현재 state가 이미 유용한 정보를 포함하고 있다. = memoryless)

Markov state {{< katex display=false >}}s{{< /katex >}}로부터 {{< katex display=false >}}s'{{< /katex >}} 으로 변경하는 transition probability 를 하는 수식은 다음과 같다.
{{< katex display=true >}}
\mathcal{P}_{ss'}  = \mathbb{P}[S_{t+1} = s'\mid S_t = s]
{{< /katex >}}
매트릭스로 표현하면 다음과 같다. (합은 1)
{{< katex display=true >}}
\mathcal{P} =  \left[
\begin{array}{ccc}
\mathcal{P}_{11} & \cdots & \mathcal{P}_{1n} \\
\vdots & \ddots & \vdots \\
\mathcal{P}_{n1} & \cdots & \mathcal{P}_{nn}
\end{array}
\right]
{{< /katex >}}

MP는 tuple로 이루어져 있다.
> A Markov Process (or Markov Chain) is a tuple {{< katex display=false >}}\langle \mathcal{S}, \mathcal{P} \rangle {{< /katex >}}
> - {{< katex display=false >}}\mathcal{S}{{< /katex >}} is a (finite) set of states  
> - {{< katex display=false >}}\mathcal{P}{{< /katex >}} is a state transition probability matrix,<br>
> {{< katex display=false >}}\mathcal{P}_{ss'} = \mathbb{P} \left[ S_{t+1} = s' \mid S_t = s \right]{{< /katex >}}

이 예제는 단순화한 state 변화를 나타내는 것이고,
MDP를 이용한 실제 사용은 훨신더 많은 state와 probability를 포함한다.

<img src="/images/rl-mp-example.png" alt="mp-example" style="width:100%;" />


## **Markov Reward Process(MRP) 란?**
### **MRP 속성**

reward가 추가가 된 것. MP 에 value judgment가 포함된 것 - 여기서 judgment 는 누적 reward가 얼마나 좋아질지 

> A Markov Rewards Process (or Markov Chain) is a tuple {{< katex display=false >}}\langle \mathcal{S}, \mathcal{P}, \textcolor{red}{\mathcal{R}, \gamma} \rangle {{< /katex >}}
> - {{< katex display=false >}}\mathcal{S}{{< /katex >}} is a (finite) set of states  
> - {{< katex display=false >}}\mathcal{P}{{< /katex >}} is a state transition probability matrix,<br>
> {{< katex display=false >}}\mathcal{P}_{ss'} = \mathbb{P} \left[ S_{t+1} = s' \mid S_t = s \right]{{< /katex >}}
> - <span style="color:red">{{< katex display=false >}}\mathcal{R}{{< /katex >}} is a reward function, {{< katex display=false >}}\mathcal{R}_s = \mathbb{E} \left[ R_{t+1} \mid S_t = s \right]{{< /katex >}} </span>
> - <span style="color:red">{{< katex display=false >}}\gamma{{< /katex >}} is a discount factor, {{< katex display=false >}}{\gamma \in [0, 1]}{{< /katex >}} </span>

timestep t에 대한 goal 은 다음과 같이 표현된다.
감마는 미래에 대한 discount factor이다. 이것이 필요한 이유는 
 - 우리는 퍼팩한 모델이 없기 때문에
 - 수학적 max 바운드(수학적 편의성을 위해)
 - MP의 무한 루프를 하지 피하기 위해
 - 근접 미래의 가치가 비근접(먼미래) 미래의 가치보다 크기 때문
 - 시퀀스의 끝이 보장된다면 discount factor를 안쓸수도 있다

{{< katex display=true >}}
G_t = R_{t+1} + \gamma R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
{{< /katex >}}

MP는 시간이 지남에 따라 확률에 의해서 state가 변경되는 것이고.<br>
MRP 에서는 그 변화된 시간에서 state에 도달할때마다 reward가 획득된다고 이해했다.<br>
이로써 현재 state 에서 바라본다면 앞으로 나의 미래 total reward를 계산할 수 있다.(discount factor 0~1)
<img src="/images/rl-mrp-example.png" alt="mp-example" style="width:80%;" />

#### **Value Function**

현재상태(s)에서의 terminated 상태에서의 Expected return 
이것은 Expectation 이다. 왜냐하면 environment는 stochastic 이니까
{{< katex display=true >}}
v(s) = \mathbb{E} [ G_t \mid S_t = s ]
{{< /katex >}}

이는 밸망방정식으로 표현될 수 있다.

#### **Bellman Equation for MRP**

Value Function은 크게 두가지 컴포넌트로 나눌수 있다.
- 현재의 리워드 {{< katex display=false >}} R_{t+1}{{< /katex >}}
- 다음계승 state의 discounted 상태 {{< katex display=false >}}\gamma v(S_{t+1}){{< /katex >}}

{{< katex display=true >}}
\begin{aligned}
v(s) &= \mathbb{E} \left[ G_t \mid S_t = s \right] \\
     &= \mathbb{E} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t = s \right] \\
     &= \mathbb{E} \left[ R_{t+1} + \gamma \left( R_{t+2} + \gamma R_{t+3} + \cdots \right) \mid S_t = s \right] \\
     &= \mathbb{E} \left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s \right] \\
     &= \mathbb{E} \left[ \textcolor{red}{R_{t+1} + \gamma v(S_{t+1})} \mid S_t = s \right]
\end{aligned}
{{< /katex >}}

{{< katex display=true >}}
v(s) = \mathbb{E} \left[ R_{t+1} + \gamma v(S_{t+1}) \mid S_t = s \right]
{{< /katex >}}

{{< katex display=true >}}
v(s) = \mathcal{R}_s + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'} v(s')
{{< /katex >}}

이를 벡터 매트릭스로 표현하면 아래와 같다.

{{< katex display=true >}}
\begin{bmatrix}
v(1) \\
\vdots \\
v(n)
\end{bmatrix}
=
\begin{bmatrix}
\mathcal{R}_1 \\
\vdots \\
\mathcal{R}_n
\end{bmatrix}
+
\gamma
\begin{bmatrix}
\mathcal{P}_{11} & \cdots & \mathcal{P}_{1n} \\
\vdots & \ddots & \vdots \\
\mathcal{P}_{n1} & \cdots & \mathcal{P}_{nn}
\end{bmatrix}
\begin{bmatrix}
v(1) \\
\vdots \\
v(n)
\end{bmatrix}
{{< /katex >}}

#### **Solving the Bellman Equation**
벨만 방정식은 linear equation 이지만 O(n^3) 복잡도를 가지고 있기 때문에 작은것만 풀수 있다.<br>
Large MRP를 풀기위해서 
 - Dynamic Programing 이나 
 - Monte-Carlo evaluation 이나 
 - Temporal-Difference Learning이 있다.

## **Markov Decision Process(MDP) 란?**

> A Markov Decision Process (or Markov Chain) is a tuple {{< katex display=false >}}\langle \mathcal{S}, \textcolor{red}{\mathcal{A}}, \mathcal{P}, \mathcal{R}, \gamma \rangle {{< /katex >}}
> - {{< katex display=false >}}\mathcal{S}{{< /katex >}} is a (finite) set of states  
> - <span style="color:red">{{< katex display=false >}}\mathcal{A}{{< /katex >}} is a (finite) set of actions </span>
> - {{< katex display=false >}}\mathcal{P}{{< /katex >}} is a state transition probability matrix,<br>
> {{< katex display=false >}}\mathcal{P}^a_{ss'} = \mathbb{P} \left[ S_{t+1} = s' \mid S_t = s, A_t = \textcolor{red}a \right]{{< /katex >}}
> - {{< katex display=false >}}\mathcal{R}{{< /katex >}} is a reward function, {{< katex display=false >}}\mathcal{R}^a_s = \mathbb{E} \left[ R_{t+1} \mid S_t = s , A_t = \textcolor{red}a \right]{{< /katex >}} 
> - {{< katex display=false >}}\gamma{{< /katex >}} is a discount factor, {{< katex display=false >}}{\gamma \in [0, 1]}{{< /katex >}} 

위 MP MRP 예제와 다르게 action이 추가됨. <br>
그림에서는 잘 안표현되어있지만, 액션을 하면. 그 액션으로인해 특정 state로 전이되는 것은 확률이다.<br>
밑에 pub에 가는것 액션을 수행하면 확률적으로 class1, class2, class3에 도달한다.
<img src="/images/rl-mdp-example.png" alt="mp-example" style="width:80%;" />

### **Policy**

정책(Policy {{< katex display=false >}}\pi{{< /katex >}}) 는 주어진 state에 대한 action의 분포
{{< katex display=true >}}
\pi(a \mid s) = \mathbb{P}[A_t = a \mid S_t = s]
{{< /katex >}}
_마크로프 속성에 의해서 현재 state는 reward를 fully characterize 한것이기 때문에 위 수식에 reward가 없다_

state 시퀀스(상태전의)에 대해서 policy를 넣으면 Markov Process이고,
state 시퀀에 Reward를 넣으면 Markov Reward process 이다.

Markov Reward Process 에 대해서 
MDP 수식으로  (Action으로부터) 다음과 같이 수식으로 표현이 가능하다. 
(MDP 수식으로(policy-action이 포함된 버전으로) MP와 MRP를 표현이 가능하다)

{{< katex display=true >}}
\mathcal{P}^\pi_{s, s'} = \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathcal{P}^{a}_{s s'} \\
\mathcal{R}^\pi_s = \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathcal{R}^a_s
{{< /katex >}}
이는 모든 action에 갈수 있는 prob를 average로({{< katex display =false >}}\pi{{< /katex >}}는 0~1의 값이므로) 이해를 쉽게하기 P와 R을 표현한 것.


MDP에 대한 value function은
state-value 펑션과, action-value function 두가지 방식이 있다.
### **state-value function**
다음과 같고 이는 **현재 state일때 {{< katex display =false >}}\pi{{< /katex >}} policy를 따를때 얼마나 좋은지**를 나타낸다. (얼마나 Reward를 얻을지)
{{< katex display = true >}}
v_\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]
{{< /katex >}}
여기서 {{< katex display =false >}}\mathbb{E}_\pi{{< /katex>}}는 모든 샘플액션에 대한 expectation


### **action-value (q) function**
이를 action-value function {{< katex display =false >}}q_\pi{{< /katex >}} 로 나타낼 수 있다.
이는 **현재 state에서 어떤action을 선택했을때 얼마나 좋은지**를 나타낸다. (얼마나 Reward를 얻을지)
{{< katex display = true >}}
q_\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right]
{{< /katex >}}


state-value function과 action-value function의 Bellman Expectation 방정식을 
다음과 같이 나타낼수 있다.
{{< katex display=true >}}
v_\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s \right]
{{< /katex >}}

{{< katex display=true >}}
q_\pi(s, a) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a \right]
{{< /katex >}}


state-value-function 와 action-value function 중 어떤 것을 중점적으로 학습하는지에 따른 학습방법이 달라지는 것 같다.<br>
**왼쪽**은 state-value function관점에서의 그림과 수식표현이고, **오른쪽**은 action-value 펑션관점에서 수식과 표현이다.
action-value function에 대해서는, action을 선택함으로써 reward를 통해서 state-value function으로 다시 넘어가는 것을 볼수 있다.

<div style="display: flex; gap: 20px;">
  <img src="/images/rl-mdp-bellman-state-value.png" alt="state-value" style="width: 50%;" />
  <img src="/images/rl-mdp-bellman-action-value.png" alt="action-value" style="width: 50%;" />
</div>

이 두개의 그래프를 합치면 다음과 같다.
왼쪽은 state-value function 관점에서의 수식이고, 오른쪽은 action-value function 관점에서의 수식이다.

<div style="display: flex; gap: 20px;">
  <img src="/images/rl-mdp-bellman-state-value-merged.png" alt="state-value-merged" style="width: 50%;" />
  <img src="/images/rl-mdp-bellman-action-value-merged.png" alt="action-value-merged" style="width: 50%;" />
</div>

이것들은 앞에서 언급했던것처럼 두 가지 Junk로 나눌수 있고, (이번 Step에서의 Reward와 미래의 value function 리턴값)
다음 수식으로 나타내진다 (matric-form).
추가로 모든 MDP는 MRP로 표현이 가능하다.
{{< katex display=true >}}
v_\pi = \mathcal{R}^\pi + \gamma \mathcal{P}^\pi v_\pi
{{< /katex >}}
{{< katex display=true >}}
v_\pi = \left( I - \gamma \mathcal{P}^\pi \right)^{-1} \mathcal{R}^\pi
{{< /katex >}}

우리는 이로부터(state-value, action-value function으로부터) **Optimal Value Function** 을 찾는다

### **Optimal Value Function**

MDP에서의 최적 행동을 찾는 방법은 optimal state-value function {{< katex display = false >}}v_*(s){{< /katex >}} 를 구하는 것이다. <br>
이것은 모든 policy 에 대해서 value function을 최대화 하는것이다. (장기적으로 최대 보상을 얻기 위해서) <br>
optimal action-value function {{< katex display = false >}}q_*(s,a) {{< /katex >}}의 경우 아래와 같이 구할 수 있다.

{{< katex display=true >}}
v_*(s) = \max_\pi v_\pi(s)
{{< /katex >}}
{{< katex display=true >}}
q_*(s, a) = \max_\pi q_\pi(s, a)
{{< /katex >}}

optimal policy ({{< katex display = false >}}\pi_*{{< /katex >}}) 는 {{< katex display = false >}}q_*{{< /katex >}} 를 최대화 함으로써 얻을 수 있다.
{{< katex display=true >}}
\pi_*(a \mid s) = 
\begin{cases}
1 & \text{if } a = \arg\max\limits_{a \in \mathcal{A}} q_*(s, a) \\
0 & \text{otherwise}
\end{cases}
{{< /katex >}}

옵티멀한 해는 위에 구한 도식과 같은 형식으로 아래 모형과 수식으로 나타낼 수 있다.
<div style="display: flex; gap: 20px;">
  <img src="/images/rl-mdp-bellman-state-value-opt-merged.png" alt="state-value-merged" style="width: 50%;" />
  <img src="/images/rl-mdp-bellman-action-value-opt-merged.png" alt="action-value-merged" style="width: 50%;" />
</div>


### **Solving Bellman Optimality Equation**
Bellman Optimality Equation은 non-linear 하고 보통 No closed form 으로 제공됨.
아래와 같은 solving method 들이 있음
- **value iteration** : Iteratively updates value estimates using the Bellman optimality equation until convergence.
- **policy iteration** : Alternates between policy evaluation and policy improvement until the policy becomes stable.
- **Q-learning** : Off-policy method that directly learns the optimal action-value function from experience.
- **Sarsa**(State-Action-Reward-State-Action) : On-policy method that updates action-values based on the action actually taken by the current policy.
 

### **Extensions of MDPs**
- infinite and continous MDPs
  - 무한한 기존 방법을 그대로 적용이 가능하다(Straightforward). 연속적인 숫자라면 편미분 해야한다.
- Partially observable MDPs
  - (finite sets of Observation:O 와 Observation function:Z) 추가요소가 있으며. 상태를 직접적으로 알 수 없으니 관측값으로부터 추정하는 hidden stage가 있는 MDP로 해결
- Undiscounted, average reward MDPs
  - ergodic markvo process 로 처리할수 있다. ergodic은 Recuurent: 각 state는 무한 시간 동안에 방문, Aperiodic : 어떤 주기성 없이 방문 하는 속성을 가지고 있다.이것은 average reward MDP이다 (discount 되지 않으니, 큰수의 법칙에 의해서). 따러서 average bellman equation 을 풀면 된다.

## **참조**
https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-2-mdp.pdf

https://www.youtube.com/watch?v=lfHX2hHRMVQ

https://trivia-starage.tistory.com/280