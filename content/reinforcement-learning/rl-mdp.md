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

거의 모든 RL 관련 문제들은 MDP로 수식화 할 수 있다(Fully observable이나 Partially ovservable이나)

Markov Property를 이용하는데, --이전 강의참조--

요약하면, 현재 state만으로 미래를 예측해도 된다는 속성이다.
(다르게 말하면, 현재 state가 이미 유용한 정보를 포함하고 있다. memoryless)

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

> A Markov Rewards Process (or Markov Chain) is a tuple {{< katex display=false >}}\langle \mathcal{S}, \mathcal{P}, \mathcal{R}, \gamma \rangle {{< /katex >}}
> - {{< katex display=false >}}\mathcal{S}{{< /katex >}} is a (finite) set of states  
> - {{< katex display=false >}}\mathcal{P}{{< /katex >}} is a state transition probability matrix,<br>
> {{< katex display=false >}}\mathcal{P}_{ss'} = \mathbb{P} \left[ S_{t+1} = s' \mid S_t = s \right]{{< /katex >}}
> - {{< katex display=false >}}\mathcal{R}{{< /katex >}} is a reward function, {{< katex display=false >}}\mathcal{R}_s = \mathbb{E} \left[ R_{t+1} \mid S_t = s \right]{{< /katex >}} 
> - {{< katex display=false >}}\gamma{{< /katex >}} is a discount factor, {{< katex display=false >}}{\gamma \in [0, 1]}{{< /katex >}} 

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

#### **Value Function**

현재상태(s)에서의 terminated 상태에서의 Expected return 
이것은 Expectation 이다. 왜냐하면 environment는 stochastic 이니까
{{< katex display=true >}}
v(s) = \mathbb{E} [ G_t \mid S_t = s ]
{{< /katex >}}

이는 밸망방정식으로 표현될 수 있다.

#### **Bellman Euation for MRP**

Value Function은 크게 두가지 컴포넌트로 나눌수 있다.
- 현재의 리워드 {{< katex display=false >}} R_(t+1){{< /katex >}}
- 다음계승state의 discounted 상태 {{< katex display=false >}}\gamma v(S_{t+1}){{< /katex >}}

{{< katex display=true >}}
\begin{aligned}
v(s) &= \mathbb{E} \left[ G_t \mid S_t = s \right] \\
     &= \mathbb{E} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t = s \right] \\
     &= \mathbb{E} \left[ R_{t+1} + \gamma \left( R_{t+2} + \gamma R_{t+3} + \cdots \right) \mid S_t = s \right] \\
     &= \mathbb{E} \left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s \right] \\
     &= \mathbb{E} \left[ R_{t+1} + \gamma v(S_{t+1}) \mid S_t = s \right]
\end{aligned}
{{< /katex >}}

{{< katex display=true >}}
v(s) = \mathbb{E} \left[ R_{t+1} + \gamma v(S_{t+1}) \mid S_t = s \right]
{{< /katex >}}

{{< katex display=true >}}
v(s) = \mathcal{R}_s + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'} v(s')
{{< /katex >}}

이를 벡터 매트릭스로 표현하면 아래와 같다.

## **Markov Decision Process(MDP) 란?**

> A Markov Decision Process (or Markov Chain) is a tuple {{< katex display=false >}}\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle {{< /katex >}}
> - {{< katex display=false >}}\mathcal{S}{{< /katex >}} is a (finite) set of states  
> - {{< katex display=false >}}\mathcal{A}{{< /katex >}} is a (finite) set of actions  
> - {{< katex display=false >}}\mathcal{P}{{< /katex >}} is a state transition probability matrix,<br>
> {{< katex display=false >}}\mathcal{P}^a_{ss'} = \mathbb{P} \left[ S_{t+1} = s' \mid S_t = s, A_t = a \right]{{< /katex >}}
> - {{< katex display=false >}}\mathcal{R}{{< /katex >}} is a reward function, {{< katex display=false >}}\mathcal{R}^a_s = \mathbb{E} \left[ R_{t+1} \mid S_t = s , A_t =a \right]{{< /katex >}} 
> - {{< katex display=false >}}\gamma{{< /katex >}} is a discount factor, {{< katex display=false >}}{\gamma \in [0, 1]}{{< /katex >}} 

### **Policy**

정책(Policy {{< katex display=false >}}\pi{{< /katex >}}) 는 주어진 state에 대한 action의 분포
{{< katex display=true >}}
\pi(a \mid s) = \mathbb{P}[A_t = a \mid S_t = s]
{{< /katex >}}
마크로프 속성에 의해서 현재 state는 reward를 fully characterize 한것이기 때문에 수식에 reward가 없다

state 시퀀스에 대해서 폴리시를 넣으면 Markov Process이고,
state 시퀀에 리워드를 넣으면 Markov reward process 이다.

마르코프 reward process 에 대해서 
MDP 수식으로  (action으로부터) 다음과 같이 수식으로 표현이 가능하다. 
(MDP 수식으로(policy-action이 포함된 버전으로) MP와 MRP를 표현이 가능하다)

{{< katex display=true >}}
\mathcal{P}^\pi_{s, s'} = \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathcal{P}^{a}_{s s'} \\
\mathcal{R}^\pi_s = \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathcal{R}^a_s
{{< /katex >}}
이는 모든 action에 갈수 있는 prob를 average로(파이는 0~1의 값이므로) 이해를 쉽게하기 P와 R을 표현한 것.


MDP에 대한 value function은
stage-value 펑션과, action-value function 두가지 방식이 있다.
### **state-value function**
다음과 같고 이는 **현재 state일때 pi 폴리시를 따를때 얼마나 좋은지**를 나타낸다. (얼마나 리워드를 얻을지)
{{< katex display = true >}}
v_\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]
{{< /katex >}}
여기세 E_pi는 모든 샘플액션에 대한 expectation


### **action-value function**
이를 action-value function q_pi 로 나타낼 수 있다.
이는 **현재 state에서 어떤떤action을 선택했을때 얼마나 좋은지**를 나타낸다. (얼마나 리워드를 얻을지)
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


state-value-function 와 action-value function 중 어떤 것을 중점적으로 학습하는지에 따른 학습방법이 달라지는 것 같다.
왼쪽은 state-value 펑션관점에서의 그림과 수식표현이고, 오른쪽은 action-value 펑션관점에서 수식과 표현이다.
action-value 펑션에 대해서는, action을 선택함으로써 reward를 통해서 state-value 펑션으로 다시 넘어가는 것을 볼수 있다.

<div style="display: flex; gap: 20px;">
  <img src="/images/rl-mdp-bellman-state-value.png" alt="state-value" style="width: 50%;" />
  <img src="/images/rl-mdp-bellman-action-value.png" alt="action-value" style="width: 50%;" />
</div>

이두개의 그래프를 합치면 다음과 같다.
왼쪽은 stae-value 펑션 관점에서의 수식이고, 오른쪽은 action-value 펑션 관점에서의 수식이다.

<div style="display: flex; gap: 20px;">
  <img src="/images/rl-mdp-bellman-state-value-merged.png" alt="state-value-merged" style="width: 50%;" />
  <img src="/images/rl-mdp-bellman-action-value-merged.png" alt="action-value-merged" style="width: 50%;" />
</div>

이것들은 앞에서 언급했던것처럼 두가지 파트로 나눌수 있고, (이번 스텝에서의 리워드와 미래의 value function 리턴값값)
다음 수식으로 나타내진다 (maxtrix-form).
추가로 모든 MDP는 MRP로 표현이 가능하다.
{{< katex display=true >}}
v_\pi = \mathcal{R}^\pi + \gamma \mathcal{P}^\pi v_\pi
{{< /katex >}}
{{< katex display=true >}}
v_\pi = \left( I - \gamma \mathcal{P}^\pi \right)^{-1} \mathcal{R}^\pi
{{< /katex >}}

우리는 이로부터(state-value, action-value function으로부터) **Optimal Value Function** 을 찾는다

### **optimal value function**

mdp에서의 최적 행동을 찾는 방법은 optimal state-value function v_*(s) 를 구하는 것이다.
이것은 모든 policy 에 대해서 value function을 최대화 하는것이다. 
optimal action-value function q_*(s,a) 의 경우 아래와 같이 구할 수 있다.

{{< katex display=true >}}
v_*(s) = \max_\pi v_\pi(s)
{{< /katex >}}
{{< katex display=true >}}
q_*(s, a) = \max_\pi q_\pi(s, a)
{{< /katex >}}

optimal policy 는 q_* 를 최대화 함으로써 얻을 수 있다.
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
- **policy interation** : Alternates between policy evaluation and policy improvement until the policy becomes stable.
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