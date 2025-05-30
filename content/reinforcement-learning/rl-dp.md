+++
date = '2025-05-29T12:21:05+09:00'
title = 'temp : 3. Planning by Dynamic Programming'
subtitle =  '강화학습의 value iteration, policy iteration과 DP 대한 내용'
weight = 6
tags = ["Definition", "value iteration", "policy iteration", "DP", "Dynamic Programming","Reinforcement Learning"]
categories = ["Reinforcement Learning"]
+++

# **3. Planning by Dynamic Programming**

{{% hint warning %}}
// NOTE: 이 페이지는 임시로 작성되었습니다.
{{% /hint %}}


MDP를 푸는 방식들이 여러 방법이 있다.

**Policy evaluation, Policy iteration, Value iteration** 등이 있고, 이것들은 **환경을 정확하게 안다면(=모델을 안다면)** DP가 적용이 가능하다 

먼저 간략하게 언급하자면 
1) **policy iteration** : policy를 평가하고 iteration 하면서 발전해나가는 방식과  (policy evaluation + policy improvement)
2) **value iteration** : value function을 iteration하면서 옵티멀을 찾아가는 방법 

이 있다.

이번 섹션은 DP로 known MDP를 푸는 방법에 대한것이고, 이것은 강화학습의 flow 와 수식 간의 이해를 위한 섹션이다.
4장부터 unknown MDP를 푸는 방법이 기술되어 있다.

## **Dynamic Programming(DP) 이란?**

**정의 : // 이부분은 생략**

MDP는 DP로 문제를 풀기에 필요한 조건들을 만족한다. 
 - 벨만 방정식(Bellman equation) 은 재귀적 decomposition
 - value function 은 값을 저장하고, 재사용한다.

<U>full environment 정보가 주어지면</U> 이것은 강화학습의 문제가 아니라 planning problem(mdp를)으로써 DP로 풀수 있다. <br>
MDP planning의 두가지 문제가 있다. (For prediction, For control)
 - **prediction problem**은 input MDP(or MRP)와 policy가 주어졌을때, 이것의 output인 value function {{< katex display=false >}}v_\pi{{< /katex >}} 을 구하는것

 - **control problem** 은 옵티마이징 하는것(best policy와 그에따른 best value function을 구하는것). <br>
input으로 MDP가 주어지고 output으로 {{< katex display=false >}}v_*{{< /katex >}} (optimal value function) 또는 {{< katex display=false >}}\pi_*{{< /katex >}} (optimal policy)

### **policy evaluation**
policy시가 얼마나 좋은지 평가(MDP로 얼마나 얼마나 많은 reward를 얻을수 있는지?). policy를 업데이트하진 않는다<br>
bellman expectation equation을 사용

bellman expectation equation을 풀기 위해서,<br>
이터레이션마다 policy하의 value function을 평가해서, value function을 업데이트 한다.<br>
{{< katex display=false >}}v1 	\rightarrow v2 	\rightarrow ... 	\rightarrow v_\pi{{< /katex >}} 가 되어 true value function 을 얻을 수 있다.

{{% hint info %}}
이것은 우리가 policy 에 따른(고정된 policy) 정확한 value function을 모르니 (optimal value function을 말하는것이 아님), 그것을 계산하기 위해서 iterative하게 계산한다는 의미. <br>
여기에서 iterative는 강화학습에서 action하고 reward 받는 iterative(timestep) 과는 다른 의미
{{% /hint %}}

이때 두가지 방식으로 backup이 가능하다. <br>
**synchronous backup/asyncronous backup** <br>
synchronous backup <br>=  한 iteration에서 전체 상태들의 값이 한꺼번에 업데이트됨 <br>
 트리에서 현재root 노드에 있다고 가정하면, 취할 수 있는 모든 action을 고려하고 갈수 있는 모든 계승 state도 고려해서 backup(되돌아가서) 현재 노드에 probability 에 따른 weight를 더한다. 이것이 결국 현재 노드의 이번 이터레이션의 value function.
 
 이것은 true value function으로 수렴하는것을 보장한다 <br>(why?= discount factor 가 0~1 이므로 수축 매핑 성질을 가진다 (contraction mapping) by GPT)

#### synchronous backup policy evaluation example
아래그림은 이동을 uniform random하기 pick된다는 policy에 대한 그림이다 (왼쪽에 적힌 숫자값들 : 1/4씩 가능성이 있는경우)
k=1 일때의, 주변 으로 갔을때 전부 -1 이니 -1*4/4 로 1회 업데이트 <br>
k=2 일때의 1.7 은 1.75가 짤린것. <br>
[0,1] 을보면 네곳으로 이동할수 있고 <br>
북으로가면 자기자신으로 돌아와서 이동-1, 이전step(k=1에서의) 자가자신의 값 -1 이므로 -2, <br> 동으로가면 이동-1 과 이전step의 [0,2]의 값 -1 이 합쳐져서 -2, 마찬가지로 남으로가면 -2 <br>
서로 가면 이동 -1 과 이전step의 [0,0]의 값 0이 합쳐져서 0 <br>
따라서 (0 -2 -2 -2) / 4 하면 [0,1] 은 -1.75 

이것을 계속하면 값이 계속 업데이트가 되는데 k=3일때 벌써 수렴한것을 볼수 있다.
<div style="display: flex; gap: 20px;">
  <img src="/images/rl-iterative-policy-evaluation.png" alt="state-value-merged" style="width: 50%;" />
  <img src="/images/rl-iterative-policy-evaluation2.png" alt="action-value-merged" style="width: 50%;" />
</div>

이 value function을 random policy가 아니라 그리디하게 선택하게 하는 policy 하면 (값이 큰것을 선택하도록하면) 오른쪽 화살표 그림과 같이 나타난다.

value function을 better 폴리시를 찾아내는데 도음을 준다.
현재 policy를 평가하는것만으로도 우리는 더좋은 새로운 폴리시를 만들수 있다.

asyncronous backup =  전체 상태를 한 번에 갱신하는 것이 아니라, 선택된 특정 상태에 대해서만 value function을 갱신하는 방식이다. 이는 계산 효율을 높이고, 빠른 수렴을 가능하게 한다. by GPT

### **policy iteration**
policy를 inner loop에서 iteration마다 평가하면서 policy가 더 나아지도록 적용해나가는 방식; 
결국 optimal policy를 찾게 된다는 설명.

첫번째 스텝으로 **policy evaluation** : policy {{< katex display=false >}}\pi{{< /katex >}} 를 평가(evaluation) 하면 value function이 나오고, (현재 policy로 각상태의 value 를 구하는것)
{{< katex display=true >}}
v_\pi(s) = \mathbb{E} \left[ R_{t+1} + \gamma R_{t+2} + \dots \mid S_t = s \right]
{{< /katex >}}
두번째 스텝으로 **policy improvement**: {{< katex display=false >}}
v_\pi(s){{< /katex >}} 기반으로 policy 를 greedily 행동하게 improvement 하면 이것이 업데이트된 policy다.
{{< katex display=true >}}
\pi' = \text{greedy}(v_\pi)
{{< /katex >}}
이것이 결국 optimal policy다. (value function도 결국 optimal한 것으로 수렴한다.)

> **추가질문:언제 수렴했는지는 어떻게 파악할 수 있는지?**
> - 정책이 더이상 바뀌지 않거나, 정책간의 차이가 아주 작을때 (10^-4) 수렴했다고 파악 by GTP
> - DQN (2015)	Validation 환경에서의 평균 reward가 더 이상 증가하지 않을때
> - PPO (2017)	평균 reward의 moving average가 안정될때 (변화량 < threshold) 
> 
> 이것들은 결국 Modified Policy Iteration. <br>
> 명시적으로 stop할 iteration 숫자(k)를 세팅하거나, 입실론-convergence of function 으로 stopping condition을 만들어야함. 

<div style="display: flex; gap: 20px;">
  <img src="/images/rl-policy-interation.png" alt="state-value-merged" style="width: 70%;" />
  <img src="/images/rl-policy-interation-eval-impro.png" alt="action-value-merged" style="width: 30%;" />
</div>


만약 policy가 deterministic 한 policy 이라면, {{< katex display=false >}} a = \pi(s){{< /katex >}}.
improvement가 멈춘다면 수식적으로 수렴한다는것을(Bellman Optimal Equation을 만족함을) 증명할 수 있지만, 생략함.
lecture-3 의 17page

### **value interation**
이것은 MDP를 푸는 또 다른 방식이다. 

bellman equation을 통해서 value function이 better 하도록 하는 방식
정책 반복보다 계산량이 적고, 수렴 속도가 빠를 수 있습니다.

어떠한 optimal policy도 두개의 컴포넌트로 나뉠수 있다.
- optimal first action {{< katex display = false >}}A_*{{< /katex >}}
- 계승되는 state S' 의 optimal policy

이를 다시말하면, <br>
현재상태에서 다음행동 (첫번째 action)이 optimal한 것을 선택하면, <br>
그다음은 계승 state S' 에서 optimal policy따르는것 <br>

이는  **Principle of Optimality** 를 나타낸다.

한 policy이 어느 한 상태에서 최적이라면(Optimal이라면), <br>
그 policy이 앞으로 갈 모든 경로에서도 계속 최적이어야 한다.

이는 아래 수식을 (value function을) 최대화 하는것으로 나타낼수 있다.
{{< katex display = true >}}
v_*(s) \leftarrow \max_{a \in \mathcal{A}} \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \, v_*(s')
{{< /katex >}}


아래그림은, 어떻게 flow가 진행되는지 직관적으로 이해하기 위한 예제이다

 syncronous 하게 업데이트가 되니 숫자가 채워져 있다.
이동시 reward는 -1 <br> 
terminate state 0인상태에서 출발. 인접 슬롯은 첫번째 iter에 -1 된다. 자세한 flow는 다음과 같다.
<br>
- V_1에서 V_2로 간다면 <br>
인접 행렬[0,1] 은 주변 자기로부터 한칸 이동한경우가 -1 =  0(왼칸 값) + -1(이동) 이 최대값이므로 취함. (네방향 모두)<br>
[0,2] 의 경우도 동일: 주변 자기로부터 한칸 이동한경우가 -1 =  0(왼칸 값) + -1(이동) 이 최대값이므로 취함. (네방향 모두)<br>
- V_2에서 V_3으로 간다면
[0,1] 의 경우 왼칸은 -1 =    0(왼칸 값) + -1(이동) <br>
나머지 방향은 -2 =   + -1(동남북 값) + -1(이동)  이므로 최대값 -1을 취함 <br>
[0,2] 의 경우 : 주변 자기로부터 한칸 이동한경우가 -2 =  1(왼칸 값) + -1(이동) 이 최대값이므로 취함. (네방향 모두)

<img src="/images/rl-value-iteration-example.png" alt="rl-essential" style="width:80%;" />

사실 우리가 모든 environment가 어떻게 동작하는지 명확하게 안다면 위에 예제의 경우, goal로 부터 인접 값들을 채워가면서 계산하면 풀리는 문제이다.

만약 우리가 syncronous dynamic programming 으로 푼다면 이게 언제 풀리는지 모른다.
(모든 state 값을 업데이트하기 때문에 v_2에서 [2,2] 의 경우 -1 로 채워져 있는데, 이게 잘채워진건가? 를 판단하지 못함)

결국 value iteration 은
optimal policy {{< katex display = false >}}\pi{{< /katex >}}를 찾는것

계속해서 value function을 업데이트하면서 최적을 찾아가고 있기 때문에
 **명시적으로 policy를 만들지 않는다.**

수식적으로는 다음과 같이 표현되어 있다.

{{< katex display = true >}}
\mathbf{v}_{k+1} = \max_{a \in \mathcal{A}} \left( \mathcal{R}^a + \gamma \mathcal{P}^a \mathbf{v}_k \right)
{{< /katex >}}


### **Synchronous Dynamic Programing**
아래 도표와 같이 처리하면 된다
<img src="/images/rl-synchronousDP-algo.png" alt="rl-essential" style="width:80%;" />

- state-value function을 base로 {{< katex display= false >}} v_\pi(s) {{< /katex >}} 나 {{< katex display= false >}} v_*(s) {{< /katex >}} 를 찾는다면 <br>
iteration당 {{< katex display= false >}} \mathcal{O}(mn^2) {{< /katex >}} 시간복잡도고 m = actions, n = states 
- action-value function 을 베이로하면 {{< katex display= false >}} q_\pi(s,a) {{< /katex >}} 나 {{< katex display= false >}} q_*(s,a) {{< /katex >}}를 찾는다면 <br>
iteration당  {{< katex display= false >}} \mathcal{O}(m^2n^2){{< /katex >}} 시간복잡도

### **Asynchronous Dynamic Programing**
위의 예제는 모든 state를 모두 업데이트 하는데 낭비가 심함.

정의: 모든 상태를 동시에 업데이트하지 않고, 일부 상태만 선택적으로 업데이트합니다.
장점: 계산 효율성이 높아지고, 특정 상태에 집중할 수 있습니다.

asynchronous dynamic programming을 하기위한 3가지 idea들은 다음과 같다
- In-place dynamic programming
- Prioritised sweeping
- Real-time dynamic programming

자세한 내용은 생략

### **Full-Width Backup 과 Sample Backup**
Full-width backup은 너무 비싸서 sample 기반 backup을 한다.

> **sample** : 에이전트가 환경과 상호작용해서 얻은 한 번의 경험 데이터

이로인한 장점은 다음과 같다.
1) 이로인해서 궁극적으로 Model-free하게 된다.( environment 모델을 알지 않아도 되게 된다)
2) 차원의 저주 해소
3) backup cost가 줄어듬

샘플을 통해서 Dynamic programing에서 model-free reinforcement learning 문제로 변환하게 된다.

> Model-based :	환경의 동작 방식을 알고 있음 (혹은 학습함). 이를 사용해 planning을 함.
> 
> Model-free	: 환경의 동작 방식 없이, 직접 환경과 상호작용하며 학습함. 오직 경험(transition, reward)에만 의존.



{{% hint warning %}}
매번 샘플은 "운 좋을 수도 있고 아닐 수도" 있지만, 충분히 많이, 반복적으로, 그리고 잘 정해진 규칙으로 업데이트하면 결국 기대값에 수렴 → 최적에 수렴. by GPT <br>
수학적으로 증명이 되었다고도 한다..
{{% /hint %}}



# **4. model-free prediction**

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

### **Monte-carlo Control**
Monde-Carlo Policy iteration 은 이전 섹션에서 설명한것

Monde-Carlo Control 은 하나씩의 episode 가 끝난후에 policy를 업데이트하는것 (episode 단위로 policy improvement)<br>
이렇게 해도 되는 이유는(수렴하는이유는) `Greedy in the Limit with Infinite Exploration` 성질을 만족하기 때문이다..

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