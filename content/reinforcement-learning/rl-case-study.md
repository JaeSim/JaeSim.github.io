+++
date = '2025-09-30T17:08:48+09:00'
title = '10. Case Study: RL in Classic Games'
weight = 13
tags = ["Definition", "Pratice", "Case Study", "self-play RL", "minimax search", "Reinforcement Learning", "RL"]
categories = ["Reinforcement Learning"]
+++


# **10. Case Study: RL in Classic Games**

이 장에서는 각각의 RL 컴포넌트/기법이 고전 게임에서 어떻게 적용되고, 어떻게 super human performance를 내는지 설명한다.
  
 
## **Game Theory**
### **Nash equilibrium**
 - 게임에서 optimal policy는 상대 player에 영향을 받는다
 - player 본인의 policy를 {{< katex display=false >}}\pi^i{{< /katex >}} 로 표현하고
 - 다른 상대 player의 policy를 {{< katex display=false >}}\pi^-i{{< /katex >}} 로 표현하면
 - 상대 play에 대해여 best respone는 {{< katex display=false >}}\pi^i_{*}(\pi^{-i}){{< /katex >}} 로 표현된다.
 - 모든 player의 policy는 다음과 같이 joint로 표현된다
{{< katex display=true >}}\pi^i = \pi^i_{*}(\pi^{-i}){{< /katex >}}
 - 이것이 Nash equilibrium 이다
   - 각각의 player들은 다른 플레이어들의 정책에 대해서 자신의 best response를 내뱉는 지점.
   - 결국 서로간의 best response로 행동하면(행동하도록 업데이트하면) 균형점을 찾는다는 내용
   - 포커를 예를 들면, 내 최적의 policy 뿐만 아니라 딜러의 최적 policy까지 학습하고, joint한 policy가 옵티멀 policy라는 예.

그동안 배워온것은 Single-Agent RL이다.
 - 다른 player는 environment의 일부로 취급해서 MDP를 푼것

Nash equilibrium 은 **fixed-point의 self-play RL**이다. 
  - 내가 다른 player가 된다고 생각해서, play 하는것
  - 서로 최적의 행동을 하면서 업데이트 해나가면 결국 고정점(fixed point)에 도달한다.
  - 각 agent에 대해서 각각의 experience를 생성한다.
  - agente들은 각자 best response를 한다.
{{< katex display=true >}}
a_{1} \sim \pi^{1}, \; a_{2} \sim \pi^{2}, \; \ldots
{{< /katex >}}
  - 이것들을 학습해내간다.





### **Two-Player Zero-Sum Games**
게임은 perfect information과 imperfect information의 경우가 있다. <br>
 perfect는 fully observed 이고, imperfect는 partially observed 인데, 강의에서는 perfect case를 위주로 다룬다. <br>

 이후 section들은 Two-Player Zero-Sum game임을 가정하고 설명한다.<br>
따라서 중요하진 않지만 간략하게 기술한다. 
 

- 두명의 플레이어가 있고, player 1= 백, player 2= 흑이다
- zero sum game이므로 두 player 의 reward의 합은 0이다
{{< katex display=true >}}
R^{1} + R^{2} = 0
{{< /katex >}}

이것을 해결하는 방법은
 - Game tree search (planning하는것)
 - Self-play reinforcement learning

 이 있다. 우리는 Self-play reinforcement learning 로 해결하고자한다.

## **Minimax Search**
### **Minimax**
 - value function의 total reward는 joint policy {{< katex display=false >}}\pi = \langle \pi^1, \pi^2 \rangle{{< /katex >}} 로부터 표현된다.
{{< katex display=true >}}
v_{\pi}(s) = \mathbb{E}_{\pi}[G_t \mid S_t = s]
{{< /katex >}}
 - minimax value function은 player 1의 policy가 max가 되고 다른 player2의  policy가 min되는 것이다. 수식적으로 보면 다음과 같다.
{{< katex display=true >}}
v_{*}(s) = \max_{\pi^1} \min_{\pi^2} \; v_{\pi}(s)
{{< /katex >}}

 - 결국 minimax는 joint policy {{< katex display=false >}}\pi = \langle \pi^1, \pi^2 \rangle{{< /katex >}} 에 대해서 minimax value를 구하게 된다.
 - 만약 옵티멀하자면 Nash equilibrium 인 지점이다.


### **Minimax Search**
 - Minimax를 tree로 해서 search 해나가면 minimax search 이다.
 <img src="/images/rl-minimax-search.png" alt="rl-minimax-search" style="width:80%;"/>
 - depth-first game tree search 로 minimax value를 찾는것이다.
 - 지금은 이진이지만 가능한 tree 전체가 보인다고 보면 될것 같다.
 - 가면갈수록 tree는 깊어지고 지수적으로 많아지기 때문에<br>
 minimax search에서 value function approximator를 적용해서 나타낼 수 있고,  {{< katex display=false >}}v(s, \mathbf{w}) \approx v_{*}(s){{< /katex >}}
 - 이를 통해 leaf nodes의 value를 estimate 할 수 있다
 - 아래 그림은 Binary feature vector 로 value function을 근사를 한 예제이다 (체스 게임)
   - x(s) 는 해당 piece가 존재하나 여부. w는 각 piece의 가치
   - 이것은 단순히 piece의 존재만 가진것을 나타내는 feature이고, position까지하면 좀더 복잡한 feature가 된다
 
 <img src="/images/rl-chess-example.png" alt="rl-chess-example" style="width:80%;"/>


 - Deep Blue와 Chinook의 예시 설명은 생략

## **Self-Play Reinforcement Learning**
Self-Play context에 기존에 배웠던 value-based RL등을 동일하게 적용이 가능하다.
 - MC는 {{< katex display=false >}}G_t{{< /katex >}} 에 대해서 value function을 업데이트하고
{{< katex display=true >}}
\Delta \mathbf{w} = \alpha \bigl(G_t - v(S_t, \mathbf{w})\bigr) \nabla_{\mathbf{w}} v(S_t, \mathbf{w})
{{< /katex >}}

 - TD(0)는 {{< katex display=false >}}v(S_{t+1}){{< /katex >}} 에 대해서 value function을 업데이트하고
{{< katex display=true >}}
\Delta \mathbf{w} = \alpha \bigl(v(S_{t+1}, \mathbf{w}) - v(S_t, \mathbf{w})\bigr) \nabla_{\mathbf{w}} v(S_t, \mathbf{w})
{{< /katex >}}

 - TD(람다)는  {{< katex display=false >}}G_t^{\lambda}{{< /katex >}} 에 대해서 value function을 업데이트한다
{{< katex display=true >}}
\Delta \mathbf{w} = \alpha \bigl(G_t^{\lambda} - v(S_t, \mathbf{w})\bigr) \nabla_{\mathbf{w}} v(S_t, \mathbf{w})
{{< /katex >}}


### **Policy Improvement with Afterstats**

우리가 deterministic한 게임이라면 옵티멀한 value function을 충분히 찾을 수 있다.<br>
왜냐하면 **afterstate** 때문. <br>
deterministic 한 게임이기 때문에, 현재 state에서 다음 계승 state로 어떻게 바뀔지 명확하게 알기 때문이다<br>
(체스로 두면 rule을 알기때문에 이동한 다음의 state[position]를 안다.)
 - {{< katex display=false >}}v_{*}(s){{< /katex >}} 를 successor state succ(s,a) 로써 표현한다면 다음과 같다.
{{< katex display=true >}}q_{*}(s, a) = v_{*}(\text{succ}(s, a)){{< /katex >}}
 - 이를 after state에 대한 min/maximising 을 적용하면 다음과 같이 표현된다
{{< katex display=true >}}
A_t = \arg\max_{a} \; v_{*}(\text{succ}(S_t, a)) \ \text{for white} \\
A_t = \arg\min_{a} \; v_{*}(\text{succ}(S_t, a)) \ \text{for black}
{{< /katex >}}



- Logistello는 Othello 게임을 정복하기 위해서 Self-Play TD를 적용했다. 
  - policy iteration 을 다음과 같이 수행<br>
  (improvement를 one-step로 하지 않고 전체 minimax tree를 이용해서 한다)
    - current policy로 self-play game을 한다
    - Monte-Carlo 로 현재 policy 를 평가한다 (regress problem이다. 누가 이길지 예측하는)
    - Greedy policy improvement를 한다
    

 - TD-Gammon 은 backgammon 을 정복하기 위해서 (서양보드 게임인데 잘 몰라도된다 ) Non-Linear Value function Approximation을 사용했다.
   - random weight로 초기화하고, self-play로 train 하였다.
   - non-linear TD learning을 다음과 같이  표현된다.
{{< katex display=true >}}
\delta_t = v(S_{t+1}, \mathbf{w}) - v(S_t, \mathbf{w}) \\
\Delta \mathbf{w} = \alpha \delta_t \nabla_{\mathbf{w}} v(S_t, \mathbf{w})
{{< /katex >}}
   - greedy로 하여 exploration 하지 않도록함 (backgammon 은 dice로 하는거라서 이미 exploration을 내재하고 있음. 명시적으로 explore 하지 않아도 모든 state를 다가봄)


## **Combining Reinforcement Learning and Minimanx Search**

### **Simple TD**
TD는 value function을 successor value로 업데이트하는 직관이다.
 <img src="/images/rl-simple-td.png" alt="rl-simple-td" style="width:80%;"/>
 - TD의 value function approximator sms v(s,w)로 다음과 같이 표현된다
 {{< katex display=true >}}
v(S_t, \mathbf{w}) \;\leftarrow\; v(S_{t+1}, \mathbf{w})
{{< /katex >}}
 - 이것은 next state의 raw value로 update하는 모양새
 - 결국 Simple TD를 적용하는건 다음과 같이 두 단계로 나눠진다
   - TD learning으로 value function을 배운다
   - 그리고 이 value function으로 minimax search를 한다 (no laerning)
{{< katex display=true >}}
v_{+}(S_t, \mathbf{w}) \;=\;  \operatorname*{minimax}_{s \in \text{leaves}(S_t)} v(s, \mathbf{w})
{{< /katex >}}


### **TD Root**
Logistello 나 TD-Gammon은 value function 학습하기 쉬워서 잘 적용되는데<br>
Chess 등의 게임에서는 잘 적용이 안됨.<br>
Search 없이 checkmate하는 전략을 찾기 어려움
 - **minimax search value로 나온 것들을 학습에 사용**하는 직관
 <img src="/images/rl-td-root.png" alt="rl-td-root" style="width:80%;"/>

 - TD root는 왼쪽 그림과 같이 search를 하고, 아 빨간색이 좋구나 하고 St+1로 이동
 - 빨간색에서 search를 해보았는데 초록색이 좋았다 라는것을 이용해서 이전 St를 업데이트
 - 수식적으로는,  다음과 같이 표현된다
   - minimax search의 값이 다음과 같이 표현될때
{{< katex display=true >}}
v_{+}(S_t, \mathbf{w}) \;=\; \operatorname*{minimax}_{s \in \text{leaves}(S_t)} v(s, \mathbf{w})
{{< /katex >}}
   - {{< katex display=false >}}l_{+}(s){{< /katex >}} 는 state s 에 대해서 minimax leaf node가 가지는 value
   - 종합하면 수식으로는 다음과 같이 표현된다.
{{< katex display=true >}}
v(S_t, \mathbf{w}) \;\leftarrow\; v_{+}(S_{t+1}, \mathbf{w}) \;=\; v(l_{+}(S_{t+1}), \mathbf{w})
{{< /katex >}}
 - 후에 기술한 TD Leaf와 다른점은 roaw value를 이용해서 업데이트한다.   



### **TD leaf**
 - 직관 
   - TD root: “지금 state → 바로 다음 state 값으로 학습”
   - TD leaf: “지금 state에서 search로 본 값 → 다음 state에서 search로 본 값으로 학습”
<img src="/images/rl-td-leaf.png" alt="rl-td-leaf" style="width:80%;"/>

 - 각각 minimax search 의 값이 다음과 같이 표현되고
{{< katex display=true >}}
v_{+}(S_t, \mathbf{w}) \;=\; \operatorname*{minimax}_{s \in \text{leaves}(S_t)} v(s, \mathbf{w}) , \
v_{+}(S_{t+1}, \mathbf{w}) \;=\; \operatorname*{minimax}_{s \in \text{leaves}(S_{t+1})} v(s, \mathbf{w})
{{< /katex >}}
 - 업데이트 대상이 St에서 minimax search했던 leaf가 된다.
{{< katex display=true >}}
v_{\textcolor{red}{+}}(S_t, \mathbf{w}) \;\leftarrow\; v_{+}(S_{t+1}, \mathbf{w}) \\
\implies \; v(l_{+}(S_t), \mathbf{w}) \;\leftarrow\; v(l_{+}(S_{t+1}), \mathbf{w})
{{< /katex >}}


### **TreeStrap**
<img src="/images/rl-treestrap.png" alt="rl-treestrap" style="width:30%;"/>

 - 그냥 imagination 한 것들로 업데이트한다는 직관
 - 수식적으로 TD Leaf와 다른것은 S_t+1 이 아닌 s인 점이다.
{{< katex display=true >}}
v(s, \mathbf{w}) \;\leftarrow\; v_{+}(s, \mathbf{w}) \\ 
\implies \; v(s, \mathbf{w}) \;\leftarrow\; v(l_{+}(s), \mathbf{w})
{{< /katex >}}
 - TD root나 TD leaf는 한번 하고 마는데, Treestrap은 search 트리 전체에서 얻은 수많은 노드와 leaf를 업데이트에 활용하기 때문에 훨씬 풍부한 학습 신호를 제공한다.


### **Simulation-Based Search**
이전강의 에서 설명했던 내용이지만 다시 다룸 <br>
self-play RL에서의 search를 simulation-based search로 대체할 수 있다는 직관 
  - AlphaZero 는 minimax의 아이디어(상대는 min, 나는 max)를 Monte Carlo 기반으로 근사

UCT는 Monte-Carlo Tree Serach인데,<br> 각 노드가 하나의 bandit으로 multi-arms bandit을 다룰때 arm을 selecting 하는것을 UCB(Upper Confidence Bounds)를 적용한 것

대부분의 게임은 simple monte-carlo search 가 거의 잘 작동한다

### **Game-Tree Serach in Imperfect information Games**
self-play의 MCTS등은 perfect information game에서는 잘 동작하나, imperfect하면 터진다. <br>
(Nash Equilibrium에 잘 수렴하지 않는다 --> player 별도 다른 observation을 가지고 있음.) <br>
궁극적으로, 해결하기위해서 smooth UCT Search 등을 소개한다.


- 예로써, 각 player가 다른 observation을 가진다고 보자. 그리고 MCTS나 다른 TreeSearch 뭐든 적용한다면 다음 그림처럼 보인다.

<img src="/images/rl-imperfect-info.png" alt="rl-imperfect-info" style="width:90%;"/>

- imperfect 하지만 여러 state가 같은 information state에 맵핑될 수 있다는 직관을 얻을 수 있다.
- 이로써 다른 player의 행동으로부터도 업데이트한다는 전략을 적용하는데 이때 업데이트를 smooth UCT Search를 사용하는 것으로 이해했다.




#### **Smooth UCT Search**

- 상대방의 현재 behavior을 학습하지 말고 average behavior을 학습한다는 직관
  - 상대방의 행동을 count하고 average 낸다.   {{< katex display=false >}}\pi_{\text{avg}}(a \mid s) \;=\; \frac{N(s,a)}{N(s)}{{< /katex >}}
   - 그렇게하여 더 robust하다 (안정적으로 학습하여 Nash Equilibrium에 도달)

- 수식적으로 UCT랑 mix한다
{{< katex display=true >}}
A \sim 
\begin{cases} 
\text{UCT}(S), & \text{with probability } \eta \\
\pi_{\text{avg}}(\cdot \mid S), & \text{with probability } 1 - \eta
\end{cases}
{{< /katex >}}