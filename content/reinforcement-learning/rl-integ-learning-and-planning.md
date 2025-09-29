+++
date = '2025-09-24T17:11:08+09:00'
title = '8. Integrating Learning and Planning'
weight = 11
tags = ["Definition", "Integrating", "Planning", "Simulation-based Search", "Monte-Carlo Tree Search", "MCTS", "Reinforcement Learning", "RL"]
categories = ["Reinforcement Learning"]
+++


# **8. Integration Learning and Planning**
 - Planning의 개념적 의미 : 모델을 알때 "앞으로 어떻게 될지?" 를 planning 할 수 있음
 
## **Model-based Reinforcement Learning**
여태까지 배운 것은 Model-free여서 모델을 학습하지 않았다.<br>
Model-based RL은 model을 experience로부터 배운다.<br> 
그리고 value function이나 policy 를 model로부터 **plan** 한다.

다음과 같은 사이클을 가지고 있다.

<img src="/images/rl-model-based-cycle.png" alt="rl-model-based-cycle" style="width:40%; display: block;margin: 0 auto;" />


model은, <br>
환경의 동작을 근사하는 방향으로 학습한다.<br>이를 통해 simulated interaction을 할 수 있다.<br>
다음 그림은 환경(지구)를 simulation 하는 model(cartoon 지구를) 표현한 것이다.

<div style="display: flex; gap: 10px; text-align: center;">
  <div style="flex: 1;">
    <img src="/images/rl-action-state-reward.png" alt="rl-rl-action-state-reward" style="width:100%;" />
    <p><strong>Model-free RL</strong></p>
  </div>
  <div style="flex: 1;">
    <img src="/images/rl-action-state-reward-model.png" alt="rl-action-state-reward-model" style="width:100%;" />
    <p><strong>Model-based RL</strong></p>
  </div>
</div>


model-based의 장점으로는 
- 모델이 좀더 compact 하게 representation 한다 (value 나 policy 대비). <br> 체스를 예로들면, state로 하나하나 표현하면 어렵지만, 말들이 어떻게 움직일지에 대한 rule만 있으면 된다. 환경에 대해서 좀더 효과적인 표현 방법
- 티쳐처럼, 슈퍼바이져 러닝을 하는 것처럼 이해할 수 도 있다. MDP를 하나씩 계산해나가는 것이 아니라 결과를 바로 계산해줄 수 있다(simulated interaction으로). MDP를 supervised learning으로 해결하는 것처럼 보인다.<br>

- 모델 uncertainty의 reason이 될 수 있다. <br>
모델이 무엇을 잘 모르는지 정확하게 알 수 있다는 의미로 이해했다(정량화가 가능). 이를 통해 효율적인 탐색이 가능하다는 장점으로 이해함.

단점으로는 
 - 모델도 학습하고 이로 인해 value function을 construction 하기 때문에 두곳에서 approximation error가 발생한다


### **Learning a Model**
모델을 수식적으로 표현하면

 - model {{< katex display=false >}}\mathcal{M}{{< /katex >}} 은 다음 MDP의 represenstaion이다. 
{{< katex display=false >}}\text{MDP} \ \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R} \rangle{{< /katex >}} 이다. 우리는 {{< katex display=false >}}\eta{{< /katex >}}로 parameterized 된다고 생각하고 아래 수식들로 표현가능하다. ({{< katex display=false >}}\eta{{< /katex >}} 는 우리가 학습할 NN의 가중치 파라미터 라고 생각하자)
 - 우리가 state {{< katex display=false >}}\mathcal{S}{{< /katex >}} 와 action {{< katex display=false >}}\mathcal{A}{{< /katex >}} 을 안다고 가정한다
 - model은 다음과 같이 표현이 가능하다
{{< katex display=true >}}
\mathcal{M} = \langle \mathcal{P}_{\eta}, \mathcal{R}_{\eta} \rangle
{{< /katex >}}
    - 여기에서 {{< katex display=false >}}\mathcal{P}_{\eta} \approx \mathcal{P}{{< /katex >}} (transition의 근사), {{< katex display=false >}}\mathcal{R}_{\eta} \approx \mathcal{R}{{< /katex >}} (reward의 근사)
 이다
    - 이를 통해 다음과 같이 표현이 가능하다
{{< katex display=true >}}
  S_{t+1} \sim \mathcal{P}_{\eta}(S_{t+1} \mid S_t, A_t)\\
  R_{t+1} = \mathcal{R}_{\eta}(R_{t+1} \mid S_t, A_t)
{{< /katex >}}

 - 보통 state와 reward가 독립적이라 가정이라서 다음과 같이 정리된다
 {{< katex display=true >}}
\mathbb{P}[S_{t+1}, R_{t+1} \mid S_t, A_t]  = \mathbb{P}[S_{t+1} \mid S_t, A_t] \, \mathbb{P}[R_{t+1} \mid S_t, A_t]
{{< /katex >}}

 - 우리의 gaol은 모델 {{< katex display=false >}}\mathcal{M}_\eta{{< /katex >}} 를 experience{{< katex display=false >}}\{ S_1, A_1, R_2, \ldots, S_T \}
{{< /katex >}} 
  로부터 estimate 하는 것이다.
 - 그러면 이것을 supervised learning problem으로 볼 수 있다. (s1에서 a1을 하면  r2을 얻으면서, s2가 되는 것이다.)
 {{< katex display=true >}}
 \begin{aligned}
S_1, A_1 &\rightarrow R_2, S_2 \\
S_2, A_2 &\rightarrow R_3, S_3 \\
&\vdots \\
S_{T-1}, A_{T-1} &\rightarrow R_T, S_T
\end{aligned}
{{< /katex >}}
   - s, a 에 대해서 r 은 regression 문제가 되고
   - s, a 에 대해서 s' 는 density estimation problem이다. <br>stochastic problem이니 s,a하면 s'로 얼마 확률로 가니? 라는 문제로 변경
   - 여기에 loss function 을 잘 선택해서 처리하면
   -  empricial loss 를 minimise 하는 파라미트 {{< katex display=false >}}\eta{{< /katex >}} 찾을 수있다.

 - 모델은 table lookup model로 할수도 있고, linear expectaion model로 할수도있고, Deep network model로 할 수도 있다.
 - 강의에서는 Table lookup model 기반으로 설명한다. (별로 효과적이진 않지만, 직관적인 이해를 위해서 그러한 것 같음)
 - 아래와 같이 table lookup 모델을 표현할 수 있다.
{{< katex display=true >}}
\hat{P}^a_{s,s'} = \frac{1}{N(s,a)} \sum_{t=1}^T 1(S_t = s, A_t = a, S_{t+1} = s') \\ 
\hat{R}^a_s = \frac{1}{N(s,a)} \sum_{t=1}^T 1(S_t = s, A_t = a) R_t
{{< /katex >}}
    - 1(⋅) 는 조건이 참이면 1, 거짓이면 0 인 함수  (indicator function)
    - N(s,a) 는 (s,a) 쌍이 얼마나 나왔는지의 대한 count
    - 즉, (s,a) 가 발생햇을대 다음 상태가 s' 인 횟수를 총 (s,a) 가 발생한 횟수로 나눈 것.
    - 나이브한 paramatic 한 접근인데 학습용으로 설명 
    - 대체제로 non-parametric 방식으로 time-step마다 experience를 tuple로 저장하고,
    {{< katex display=false >}}\langle S_t, A_t, R_{t+1}, S_{t+1} \rangle{{< /katex >}}
    <br> 주어진 (s,a)와 매칭되는 tuple을 random하게 뽑아 쓰는 방식도 있다. [보통 이걸 많이 쓰는것으로 이해 : DQN]
    {{< katex display=false >}}\langle s, a, \cdot, \cdot \rangle{{< /katex >}}

### **Planning with a Model**
위에서 만든 Model을 가지고 planning을 해보자. Planning은 MDP를 푸는 것이다.
 -  모델이 주어 졌을때, {{< katex display=false >}}\mathcal{M}_\eta = \langle \mathcal{P}_{\eta}, \mathcal{R}_{\eta} \rangle{{< /katex >}}
- MDP {{< katex display=false >}}\text{MDP} \ \langle \mathcal{S}, \mathcal{A}, \mathcal{P}_\eta, \mathcal{R}_\eta \rangle{{< /katex >}}  를 풀어야한다.
 - value interation, policy interation, tree search등 우리는 쓸 수 있다.
 - 강의에서는 sample-based planning을 다름 (심플하지만 강력)
    - 모델은 단순히 sample을 만들어내기만 한다는 직관
    - 모델의 샘플 생성은 다음과 같은 수식에 따름. 이때의 {{< katex display=false >}}S_{t+1}{{< /katex >}} 나 {{< katex display=false >}}R_{t+1}{{< /katex >}} 는 다음과 같음

{{< katex display=true >}}
S_{t+1} \sim \mathcal{P}_\eta(S_{t+1} \mid S_t, A_t) \\ 
R_{t+1} = \mathcal{R}_\eta(R_{t+1} \mid S_t, A_t)
{{< /katex >}}
 - 이후에 이 sample로부터 model-free RL을 적용(Monte-Carlo control, Sarsa, Q-learning등)
 - real experience 대비 computation 만 확보된다면 많은 sampled experience를 만들어낼 수 있다.
   - real experience로 model을 학습하고, model 이 sample experience 를 만드고, 이것을 model-free RL을 한다
   - interation이 돌수록 model이 정교해지고, sample도 정교해지고, value function등도 정교해진다.
 - model이 부정확하다면, 어떻게 해야할까? {{< katex display=false >}}\langle \mathcal{P}_\eta, \mathcal{R}_\eta \rangle \neq \langle \mathcal{P}, \mathcal{R} \rangle{{< /katex >}}
    - model-based RL은 model 이 정확해야하지 좋다
    - solution1: model-free RL 을 쓴다
    - solution2: model의 불확실성을 (uncertainty)를 명시적으로 추론(reason) 한다 

## **Integrated Architectures**
 - experience는 two source다. 
   - real experience(sampled from environment) [true MDP]
{{< katex display=true >}}
S' \sim \mathcal{P}^{a}_{s,s'}\\
R = \mathcal{R}^{a}_{s}
{{< /katex >}}
   - simluated experience(sampled from model) [approximate MDP]
{{< katex display=true >}}
S' \sim \mathcal{P}_{\eta}(S' \mid S, A)\\
R = \mathcal{R}_{\eta}(R \mid S, A)
{{< /katex >}}
 - 윗장에서 설명한 Model-Based RL은 simulated experience를 쓰는것
 - 이번 장에서 설명하는 것은 두개를 다 쓰는것
   - model을 real experience로부터 학습하고
   - value function(and/or policy)을 real 과 simulated experience를 둘다 사용한다.

<img src="/images/rl-model-based-dyna-cycle.png" alt="rl-model-based-dyna-cycle" style="width:40%; display: block;margin: 0 auto;" />

 - Dyna-Q algorithm
{{< katex display=true >}}
\begin{aligned}
&\text{Initialize } Q(s,a) \text{ and } Model(s,a) \; \forall \; s \in \mathcal{S}, a \in \mathcal{A}(s) \\
&\text{Do forever:} \\
&\quad (a)\; S \leftarrow \text{current (nonterminal) state} \\
&\quad (b)\; A \leftarrow \varepsilon\text{-greedy}(S, Q) \\
&\quad (c)\; \text{Execute action } A; \text{ observe } R, S' \\
&\quad (d)\; Q(S,A) \leftarrow Q(S,A) + \alpha \Bigl[ R + \gamma \max_a Q(S',a) - Q(S,A) \Bigr] \quad \text{// normal Q-learning update}\\
&\quad (e)\; Model(S,A) \leftarrow R, S' \quad (\text{assuming deterministic environment}) \quad \text{// model update by supervised learning}\\
&\quad (f)\; \text{Repeat } n \text{ times:} \quad \text{// sample and Q-learning}\\
&\qquad S \leftarrow \text{random previously observed state} \\
&\qquad A \leftarrow \text{random action previously taken in } S \\
&\qquad R, S' \leftarrow Model(S,A) \\
&\qquad Q(S,A) \leftarrow Q(S,A) + \alpha \Bigl[ R + \gamma \max_a Q(S',a) - Q(S,A) \Bigr]
\end{aligned}

{{< /katex >}}
 - 적용시에 RL이 스스로 어떨지 생각해서 하니, 더 빠르게 수렴한다 lecture8 28-30 page
   - page 29는 학습중에 더 어려운 환경에 대해서 학습할때의 reward를 보여준것이고, 30은 학숩중에 더 쉬운 환경에 대해서 어떻게 reward가 변화하는지를 보여준다.
   - 둘다 좀더 탐험적일때 성능이 증가한다. (쉽다->어렵다 일떄 좀더 극단적으로 나타남)

## **Simulation-based Search**
이 section은 model-based RL안에서, <br>
어떻게 더 효율적이게 planning을 할 수 있는지 관점에 대한 것을 다룬다

 - key idea는 **sampling** and **forward search** 다

 - forward search algorithm
   - 전체 state 를 탐험(explore) 하지 않는다.
   - short-term future에 어떤일이 일어날지만 집중한다.
   - ***전체 MDP를 풀지 않아도 된다*** 는 직관
      - "전체 상태 공간을 직접 학습하기는 너무 비싸다. 대신, 유망한 부분만 집중적으로 시뮬레이션하면서 거기서 경험한 정보로 Q값을 점점 더 정확하게 만든다."

<img src="/images/rl-forward-search.png" alt="rl-forward-search" style="width:80%;"/>

// Note: balsa simulation learning 과는 개념이 조금 다름

  - model에 넣어서, 다음에 무슨 state가 될지 simulation 하는것 <br>
    [ 이전 섹션에서는 model을 sample 생성기로 사용했었음 ] 
  - {{< katex display=false >}}\{ \textcolor{red}{s_t^k}, A_t^k, R_{t+1}^k, \ldots, S_T^k \}_{k=1}^K \sim \mathcal{M}_\nu{{< /katex >}}
  - 이 simulated spisode에 대해서 model-free RL을 적용한다
    - Monte-Cralo control을 적용하면 Monte-Carlo Search 라고 부른다. 
    - Sarsa 를 적용하면, TD search

### **Simple Monte-Carlo Search**
 - 모델 {{< katex display=false >}}\mathcal{M}_\nu{{< /katex >}} 가 주어졌다고 가정하면(누군가 우리에게 model을 주었다),<br>또한 simulation policy {{< katex display=false >}}\pi{{< /katex >}}가 있을때,
 - {{< katex display=false >}}a \in \mathcal{A}{{< /katex >}} 인 모든 a를 돌린다.
   - K번째 epsiode를 현재 state s_t에 대해서 simulation 하면 다음과 같이 표현된다
     - {{< katex display=false >}}s_t, a{{< /katex >}} 일때 다음,다다음,,, 것은 어떤 episode가 될지 시뮬레이션 하게 됨
   {{< katex display=true >}}
   \{ \textcolor{red}{s_t, a}, R_{t+1}^k, S_{t+1}^k, A_{t+1}^k, \ldots, S_T^k \}_{k=1}^K \sim \mathcal{M}_\nu, \pi
   {{< /katex >}}
   - 이것을 mean return 으로 action들을 evaluate한다 (Monte-Carlo evaluation) <br>
   [시뮬레이션을 돌려서 얼마나 도달할지를 계산하고, 그것을 mean한다. ]
   {{< katex display=true >}}
   Q(\textcolor{red}{s_t, a}) = \frac{1}{K} \sum_{k=1}^K G_t \quad \xrightarrow{P} \quad q_\pi(s_t, a)
  {{< /katex >}}
   - 가장 q value가 높은 것을 택한다
   {{< katex display=true >}}a_t = \arg\max_{a \in \mathcal{A}} Q(s_t, a){{< /katex >}}
   

### **Monte-Carlo Tree Search**
{{% hint info %}}
***매우 중요***<br>
강의에서는 핵심적인 직관만 말하고 넘어가는데, 정확한 동작원리에 대해서 추가적인 학습이 필요함<br>
추후 추가 업데이트 요망
{{% /hint %}}


 - 동일하게 {{< katex display=false >}}\mathcal{M}_\nu{{< /katex >}} 가 주어졌을때,
 - current simulation policy {{< katex display=false >}}\pi{{< /katex>}}로  K개 episodes들을 simulation 한다

{{< katex display=true >}}
\{ \textcolor{red}{s_t}, A_t^k, R_{t+1}^k, S_{t+1}^k, \ldots, S_T^k \}_{k=1}^K \sim \mathcal{M}_\nu, \pi
{{< /katex >}}

 - 그리고 search tree를 만든다 (visit했던 state와 action에 대한)
 - 다음 수식과 같이 Q 펑션을 evaluate하게 된다. (mean return)
 {{< katex display=true >}}
Q(\textcolor{red}{s, a}) = \frac{1}{N(s,a)} \sum_{k=1}^K \sum_{u=t}^T 1(S_u, A_u = s, a) G_u \xrightarrow{P} q_\pi(s,a)
{{< /katex >}}
 - 그리고 가장 높아지는 것을 택한다.
 {{< katex display=true >}}a_t = \arg\max_{a \in \mathcal{A}} Q(s_t, a){{< /katex >}}
 - 결국, 이전 section과 거의 같은데, simulation tree가 있어서 이를 이용해서 simulation policy {{< katex display=false >}}\pi{{< /katex >}}를 더 improve 한다.
 - 그런데, 우리는 simulation 하는 것이라서, in-tree와 out-tree로 나누어 생각한다 <br>
 현재까지 탐색된 노드라면 in-tree 아니라면 out-tree(새롭게 확장되는 트리)
   - in-tree는  Tree Policy를 사용한다 : 강의에서는 Q(S,A) 를 maxmise하는 action을 고른다,
   - out-of-tree는 Default Policy(rollout policy)를 사용한다 : 강의에서는 randomly 고른다
 - 각 시뮬레이션마다
   - Monte-Carlo evaluation으로 Q(S,A)를 평가하고
   - tree policy를 {{< katex display=false >}}\epsilon{{< /katex >}}-greedy(Q) 로 업데이트한다
    
 - 그리고 이 simluated experience를 Monte-Carlo control 적용한다
 - 그러면 optimal search tree  {{< katex display=false >}}Q(S, A) \rightarrow q_{*}(S, A){{< /katex >}}  에 converge 한다


<img src="/images/rl-mcts.png" alt="rl-mcts" style="width:40%; display: block;margin: 0 auto;" />

MCTS의 장점은
 - highly selective best-first search
   - 유망한 결로를 우선적으로 탐색하는 방법 이라는 설명 [exploration vs exploitation] : BFF, DFS, DP 등 대비
 - DP 와 다르게 state를 dynamically evaluate 한다
 - sampling을 통해서 차원의 저주 해소 [모든 가능한 environemnt를 고려하지 않아도 된다.]
 - "black-box" 에도 sample로도 동작하는 method
 - computationally efficient, paralleisable, scalability


### **TD Search**
 - 이전 강의들에서 배운 bootstrapping을 위해서 Monte-Calro 대신 TD learning을 적용할 수 있다.
 - MCTS 는 MC control를 통해서 현재 로부터의 sub-MDP를 푸는것이고, <br>
 TD search는 Sarsa 를 통해서 현재 로부터의 sub-MDP를 푼다
 - TD Learning은 variance를 줄이고 bias를 늘리고, more efficient 하다. TD Search도 동일
 - TD Search도 동일하게 TD(람다) 적용가능 (동일한 효과)
 - 시뮬레이션의 each step에 대해서 action-value(Q value)를 Sarsa로 업데이트 한다.
{{< katex display=true >}}
\Delta Q(S, A) = \alpha \bigl(R + \gamma Q(S', A') - Q(S, A)\bigr)
{{< /katex >}}
 - 그리고 action-value Q(s,a)를 base로 {{< katex display=false >}}\epsilon{{< /katex >}}-greedy 등으로 action을 선택한다. 
 - Q function에 여러 approximation을 적용해도 된다.


TD Learning과 TD Search를 합친 Dyna-2가 있다.
 - long-term memory와 short-term memory가 있다.
 - long-term memory는  real experience로부터 업데이트한다. TD Learning
   - general domain knowledge를 전체 episdoe에 적용
 - short-tern memory는 simulated experience로 부터 업데이트한ㄷ. TD Search
   - 특수한 local knowledge를 현재 상황에 적용
 - 전체 value function은 long and short-tern memory의 합
