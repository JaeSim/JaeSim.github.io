+++
date = '2025-09-23T13:22:02+09:00'
title = '7. Polciy Gradient Methods'
weight = 10
tags = ["Definition", "Model Free", "Value Function", "policy gradient", "Reinforcement Learning", "RL"]
categories = ["Reinforcement Learning"]
+++

# **7. Policy Gradient Methods**

이전섹션에서는 action-value function(state value function, action-value function)을 근사해서 옵티멀한 value function을 찾아 갔다. 이때 policy는 {{< katex display =false >}}\epsilon{{< /katex >}}-greedy. 


policy gradient는 policy parameter를 직접 업데이트 한다. **policy-based !** <br>
policy를 direct 로 업데이트 하는것 <br>
이는 large complicated environment에 좀더 적합() 각 state를 명확하게 구분하기 어려울때). <br>
 Policy는 value 대비 더 compact하게 저장한다. 저게 저장하는 장점이 있긴한다.

다음과 같은 장점이 있다고 언급한다.
- value-based 보다 convergence가 더 잘된다는 것이 있고
- high-dimensional 이나 contiunous action space에서 효과적이다.
  - 이경우, value-based는 심플한 직관이지만, 계산해야할 것이 많다
- stochastic polices를 배울 수 있다.

단점은 다음과 같다.
 - 나이브 policy는 local opitimum에 가거나 
 - slow 하게 converge 되고, high variance를 가진다.


policy에 대하여 Deterministic policy(결정적 정책) 으로 할수도 있고 stochastic policy(확률적 정책)로 할수 있다. <br>
 그러나 Deterministic하면 goal에 절대 도달을 하지 못하거나, 매우 오래걸린다. 보통 stochastic이 더 좋다..

따라서, policy는 probability 이며 현재 state에서 action을 선택할 확률 이고 action들이니 확률 분포이다.
{{< katex display =true >}}
\pi_\theta(s,a) = \mathbb{P}[a \mid s, \theta]
{{< /katex >}}
<img src="/images/rl-value-policy-diagrampng.png" alt="policy, value and action cricit" style="width:80%;"/>


 - Value-Based and Policy-based RL은 다음과 같은 특성을 가지고 있다.

| Method            | Value Function 여부      | Policy 여부         | 
|----------------|-------------------------------|----------------------------|
| **Value-Based** | Learnt Value Function   | Implicit Policy<br> (e.g. {{< katex display =false >}}\epsilon{{< /katex >}}-greedy)      |
| **Policy-Based**| No Value Function    | Learnt Policy        |
| **Actor-Critic**| Learnt Value Function   | Learnt Policy        |

 

## **Policy Objective Function**
- policy를 최적화 하려면 {{< katex display =false >}}\pi_\theta(s, a){{< /katex >}} 로 표현하고, 파라미터 {{< katex display =false >}}\theta{{< /katex >}} 에 대해서 best  {{< katex display =false >}}\theta{{< /katex >}} 를 찾아야한다 
  - 어떻게 policy {{< katex display =false >}}\pi_\theta{{< /katex >}} 를 측정하는지 세가지 방법이 있다
  - start value를 사용하는 것 {{< katex display =false >}}J_1(\theta) = V^{\pi_\theta}(s_1) = \mathbb{E}_{\pi_\theta}[v_1]{{< /katex >}}

  - average value를 사용하는 것 {{< katex display =false >}}J_{avV}(\theta) = \sum_s d^{\pi_\theta}(s) V^{\pi_\theta}(s){{< /katex >}}
  - average reward per time-step을 사용하는 것 {{< katex display =false >}}J_{avR}(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(s,a) \mathcal{R}^a_s{{< /katex >}}
  - 이것들은 결국 distrubution term이 다를뿐 같은 gradient method를 쓴다.

  - 우리는 {{< katex display =false >}}J(\theta){{< /katex >}} 를 maximise하는 {{< katex display =false >}}\theta{{< /katex >}}를 찾아야한다
    - gradient 을 접근할수 없을때 쓸수 있는 방법들이 있긴하다 (Hill Climbing, Simplex 기타 등등). 자세히 다루진 않음<br>
   강의에서는 gradient에 접근이 가능하다고 보고, gredient descent에 대해서만 설명


- 파라미터 {{< katex display =false >}}\theta{{< /katex >}} 의 델타와 목적 함수와의 관계는 다음같고.  {{< katex display =false >}}\alpha{{< /katex >}} 는 batch-size
{{< katex display =true >}}
\Delta \theta = \alpha \nabla_\theta J(\theta)
{{< /katex >}}
 - 미분된 목적함수는 다음과 같이 표현된다.
{{< katex display =true >}}
\nabla_\theta J(\theta) =
\begin{pmatrix}
\frac{\partial J(\theta)}{\partial \theta_1} \\
\vdots \\
\frac{\partial J(\theta)}{\partial \theta_n}
\end{pmatrix}
{{< /katex >}}

## **Finite Difference Policy Gradient**

 - gradient ascent 를 구한다. (reward가 최대화 되어야 하므로)
 - simplelest naive approach (MC나 actor critic 대비). numarically approach
 - k개의 dimension이 있다면, {{< katex display =false >}}\theta{{< /katex >}} 에 작은 변화량 {{< katex display =false >}}\epsilon{{< /katex >}}을 주어 (perturb 이라고 쓰네..)<br>
   각각의 변수가(k개) objective function(목적함수)의 {{< katex display =false >}}\theta{{< /katex >}}에 대해 미분량을 계산한다.
   - 그런데 이건 차원(k)이 무지하게 늘어나면 그만큼 미분 계산이 필요. awful compuation, and noisy
   - 그러나 심플하고 결국 correct direction을 찾아가게 된다
  {{< katex display =true >}}
  \frac{\partial J(\theta)}{\partial \theta_k} \approx \frac{J(\theta + \epsilon u_k) - J(\theta)}{\epsilon}
{{< /katex >}}
 
## **Monte-Carlo Policy Gradient**
 - policy gradient를 analytically 한다.
 - 우리가 gradient {{< katex display =false >}}\nabla_\theta \pi_\theta(s,a){{< /katex >}} 를 안다고 가정한다.
 - 미분을 log미분으로 다음과 같이 변형이 가능하고 ( Likelihood Ratio Trick 이라고 하는데,, 기대값의 기울기를 계산할때 자주 쓰는 기법이래)
 {{< katex display =true >}}
\nabla_\theta \pi_\theta(s,a)
= \pi_\theta(s,a)\frac{\nabla_\theta \pi_\theta(s,a)}{\pi_\theta(s,a)}
= \pi_\theta(s,a)\nabla_\theta \log \pi_\theta(s,a)
{{< /katex >}}
 - 결국 score function은   {{< katex display =false >}}\nabla_\theta \log \pi_\theta(s,a){{< /katex >}} 다
 - 이 score function은 computation이 원본보다 easy하다

### **Softmax Policy**
- parameter 와 linear combination 으로  {{< katex display =false >}}\phi(s,a)^\top \theta{{< /katex >}} 로 나타내고
- 이것을 exponentiate 하여 확률적으로 표현하고 {{< katex display =false >}}\pi_\theta(s,a) \propto e^{\phi(s,a)^\top \theta}{{< /katex >}}
- 그것을 gradient 하는 계산 식에 넣어서, 기대값으로 표현하면 다음과 같다.
  - softmax이므로 이렇고, 양변에 log 취하고 양변을 미분하면 최종 식이 된다.
{{< katex display =true >}}
\pi_\theta(a|s) = \frac{e^{\phi(s,a)^\top \theta}}
{\sum_{a'} e^{\phi(s,a')^\top \theta}}
\\
\log \pi_\theta(a|s) 
= \phi(s,a)^\top \theta 
- \log \sum_{a'} e^{\phi(s,a')^\top \theta}
\\
\nabla_\theta \log \pi_\theta(s,a) = \phi(s,a) - \mathbb{E}_{\pi_\theta}[\phi(s,\cdot)]{{< /katex >}}

직관적으로, **실제값**을 **다른 모든행동의 기대값** 에서 뺀것이 ***gradient*** 다.

### **Gaussian Policy**
 - linear combination 의 feature의 결합으로 mean을 나타냄. {{< katex display =false >}}\mu(s) = \phi(s)^\top \theta{{< /katex >}}
 - {{< katex display =false >}}\sigma^2{{< /katex >}} 도 parameterize 할수 있긴한데, 일단 심플 버전은 normal distribution일때
{{< katex display =false >}} a \sim \mathcal{N}(\mu(s), \sigma^2){{< /katex >}}

 - 최종적으로 score function은 다음과 같이 된다. 수식적으로 몇단계 건너띄긴했는데, 생략함..
 {{< katex display =true >}}
\nabla_\theta \log \pi_\theta(s,a) = \frac{(a - \mu(s)) \phi(s)}{\sigma^2}
{{< /katex >}}

 - 직관적으로 즉, 평균보다 큰 행동은 확률을 늘리고, 작은 행동은 확률을 줄이는 방향으로 파라미터 {{< katex display =false >}}\theta{{< /katex >}}가 업데이트. 결과적으로 정책은 보상 높은 행동 쪽으로 평균을 이동하게 됨.


### **Policy Gradient Theorem**
 - One-step MDP를 먼저 설명하고, 이를 확장하여 Multi-step MDP에 대해 설명한다

 **one-step MDP**에 대해서,,
 - one-step MDP란 action을 고르고, 다음단계에 reward를 얻고 termiante 되는 MDP. 이해를 위해 가정한 MDP
 - one-step MDP의 목적함수는 다음과 같고, {{< katex display =false >}}J(\theta) = \mathbb{E}_{\pi_\theta}[r] = \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_\theta(s,a) \mathcal{R}_{s,a}
{{< /katex >}}
 - Likelihood Ratio Trick 을 넣으면 다음과 같다.
{{< katex display =true >}}
\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_\theta(s,a) \nabla_\theta \log \pi_\theta(s,a) \mathcal{R}_{s,a}\\
= \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s,a) r]
{{< /katex >}}
 - 보상 r이 크면 정책 파라미터({{< katex display =false >}}\theta{{< /katex >}})를 증가시키는 것을 수식적으로 알수 있다.


**multi-step MDP**에 대해서,,
 - multi-step MDP는 one-step MDP의 imemdiate reward를 value function으로 대체한것
 {{< katex display =true >}}
 \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s,a) \, Q^\pi(s,a) \right]
 {{< /katex >}}

결국, **Policy Gradient Theorem**은 다음과 같다.
 - 모든 미분가능한 policy {{< katex display =false >}}\pi_\theta(s,a){{< /katex >}} 에 대해서,
 - 모든 목적함수, start_value를 쓰던 average reward를 쓰던,  ({{< katex display =false >}}J_1{{< /katex >}}, {{< katex display =false >}}J_{avR}{{< /katex >}} or {{< katex display =false >}}\frac{1}{1-\gamma} J_{avV}{{< /katex >}}
)
 - policy graident는 multi-step MDP 수식과 같다.
 {{< katex display =false >}}
 \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s,a) \, Q^\pi(s,a) \right]
 {{< /katex >}}

### **REINFORCE**
 - Monte-Carlo Policy Gradient의 대표적인 구현체
 - 원래라면 {{< katex display =false >}}
 \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s,a) \, Q^\pi(s,a) \right]
 {{< /katex >}} 를 구해야하는데, 어려우니
  
 - {{< katex display = false >}}Q^\pi(s,a){{< /katex >}} 의 unbiased sample (충분히 많이 구하면 Estimated 된거랑 같음[대수의 법칙]) 를  {{< katex display = false >}}v_t{{< /katex >}} 로 씀
 {{< katex display =true >}}
 \Delta \theta_t = \alpha \nabla_\theta \log \pi_\theta(s_t, a_t) v_t
 {{< /katex >}}
 - pseudo-code

 {{< katex display =false >}}
\textbf{function REINFORCE} \\
\quad \text{Initialise } \theta \text{ arbitrarily} \\
\quad \textbf{for} \text{ each episode } \{s_1, a_1, r_2, \ldots, s_{T-1}, a_{T-1}, r_T\} \sim \pi_\theta \ \textbf{do} \\
\quad \quad \textbf{for } t = 1 \ \text{to } T-1 \ \textbf{do} \\
\quad \quad \quad \theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(s_t, a_t) v_t \\
\quad \quad \textbf{end for} \\
\quad \textbf{end for} \\
\quad \textbf{return } \theta \\
\textbf{end function}
 {{< /katex >}}

 - ***직관*** <br> 
  각각의 샘플된 에피소드에 대해서, 전체 trajectory(출발부터 끝까지)를 본다음에, 각각의 Step에 대해서 gradient를 업데이트

 - 더 많은 iteration 이 value-based 보다 필요

## **Actor-Critic Policy Gradient**

policy-based + value-based

 - Monte-Carlo policy gradient는 high-variance 를 가지고 있음
 - value function approximator를 이용해서 Critic 모델을 구현. 하여 plug in
 - original policy gradient 는 다음과 같다 
{{< katex display =true >}}
 \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s,a) \, Q^\pi(s,a) \right]
 {{< /katex >}}
 - 여기에서 Q-function을 다음과 같이 근사로 표현이 가능하다 {{< katex display =false >}}Q_w(s,a) \approx Q^{\pi_\theta}(s,a){{< /katex >}}
 - 이걸 원래 수식에 넣으면 다음과 같고
 {{< katex display =true >}}
\nabla_\theta J(\theta) \approx 
\mathbb{E}_{\pi_\theta}\!\left[\,\nabla_\theta \log \pi_\theta(s,a)\; Q_w(s,a)\,\right]
{{< /katex >}}
 - 미분량은 다음과 같이 볼 수 있다.
{{< katex display =true >}}
\Delta \theta = \alpha\, \nabla_\theta \log \pi_\theta(s,a)\; Q_w(s,a)
{{< /katex >}}

 - ***직관*** <br> 
  policy-model 이 actor 가 된다. action을 선택하는. value-model이 critic이 된다. 얼마나 좋을지 등을 추정해 평가해준다 <br>
  critic이 평가해준 방향으로 업데이트가 된다. 
 - value function approximator 에 6장에서 배운 MC, TD, TD람다 등을 적용이 가능하다


### **Action-Value Actor-Critic (QAC)**
다음은 linear TD(0)를 approximator로 하였을때를 보여준다. Action-Value Actor-Critic (QAC)
 - action-value (q-value) 펑션을 다음과 같이 사용하고 {{< katex display =false >}}Q_w(s,a) \;=\; \phi(s,a)^\top w{{< /katex >}}
 - 전체 pseudo-code는 다음과 같다


{{< katex display =false >}}
\textbf{function QAC}\\
\quad \text{Initialise } s,\ \theta\\
\quad \text{Sample } a \sim \pi_\theta(\cdot \mid s)\\
\quad \textbf{for each step do}\\
\quad \quad \text{Sample reward } r = \mathcal{R}^{a}_{s};\ \text{sample transition } s' \sim \mathcal{P}^{a}_{s}\\
\quad \quad \text{Sample action } a' \sim \pi_\theta(\cdot \mid s')\\
\quad \quad \delta \;=\; r \;+\; \gamma\, Q_w(s',a') \;-\; Q_w(s,a) \text{    // critic error calculate}\\
\quad \quad \theta \leftarrow \theta \;+\; \alpha\, \nabla_\theta \log \pi_\theta(a \mid s)\, Q_w(s,a)  \text{    // actor update}\\
\quad \quad w \leftarrow w \;+\; \beta\, \delta\, \phi(s,a) \text{  // ciritic update}\\
\quad \quad a \leftarrow a',\quad s \leftarrow s'\\
\quad \textbf{end for}\\
\textbf{end function}
{{< /katex >}}


### **Advantge Actor-Critic (A2C)** 
Baseline을 이용해서 variance을 줄이는 방법
 - baseline {{< katex display =false >}}B(s){{< /katex >}} 가 있다고 하면 
    - baseline은 s,a 와 영향이 없어서 밖으로 나올수 있고, {{< katex display =false >}}\sum_{a\in\mathcal{A}} \pi_\theta(s,a){{< /katex >}}의 sum은 1이다.
    - 남은 앞의 term은 미분이므로({{< katex display =false >}}\nabla_\theta{{< /katex >}}) 항상 0이 된다
{{< katex display =true >}}
\begin{aligned}
\mathbb{E}_{\pi_\theta}\!\left[ \nabla_\theta \log \pi_\theta(s,a)\, B(s) \right]
&= \sum_{s\in\mathcal{S}} d^{\pi_\theta}(s) \sum_{a\in\mathcal{A}} \nabla_\theta \pi_\theta(s,a)\, B(s) \\
&= \sum_{s\in\mathcal{S}} d^{\pi_\theta}(s)\, B(s)\, \nabla_\theta \sum_{a\in\mathcal{A}} \pi_\theta(s,a) \\
&= 0
\end{aligned}
{{< /katex >}}
 - 좋은 baseline를 state value function을 쓰는거다. {{< katex display =false >}}B(s) = V^{\pi_\theta}(s){{< /katex >}} 
 - 그러면 우리는 q-value 와 state-value를 advantage를 쓸수 있다. {{< katex display =false >}}A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)
{{< /katex >}} 
 - 전체 수식은 다음과 같이 된다
{{< katex display =true >}}
 \nabla_\theta J(\theta)
= \mathbb{E}_{\pi_\theta}\!\left[\,\nabla_\theta \log \pi_\theta(s,a)\; A^{\pi_\theta}(s,a)\,\right]
{{< /katex >}} 
 - 직관은<br> 
  Q-value 에서 state-value 을 뺀것, 현재에서 어떤 action을 고르는게 현재 상태보다 어떤 어드벤테이지가 얼마나 있는지를 업데이트

 - 그러면 Q-value, state-value 두개를 각각 학습해야하는데, 하나만 하는 방법도 있다.

 - true value function // 가 있으면 TD error는 다음과 같이 표현된다
 {{< katex display =true >}}
\delta^{\pi_\theta} \;=\; r \;+\; \gamma\, V^{\pi_\theta}(s') \;-\; V^{\pi_\theta}(s)
{{< /katex >}} 
 - TD error이 expectation는 결국 TD error 는 advandage function의 unbiased sample이다 
 {{< katex display =true >}}
\begin{aligned}
\mathbb{E}_{\pi_\theta}\!\big[\delta^{\pi_\theta}\mid s,a\big]
&= \mathbb{E}_{\pi_\theta}\!\big[r + \gamma V^{\pi_\theta}(s') \mid s,a\big] - V^{\pi_\theta}(s) \\
&= Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s) \\
&= A^{\pi_\theta}(s,a)
\end{aligned}
{{< /katex >}} 
 - TD error에 대한 approximate하면 다음과 같이 표현이 된다.
 {{< katex display =true >}}
  \delta_{w} \;=\; r \;+\; \gamma\, V_{w}(s') \;-\; V_{w}(s)
{{< /katex >}} 
 - 그래서 Q-value를 구하지 않고도 state-value function만으로 계산할 수 있다. 우리는 parameter w에 대해서만 구하면 된다.
 - 직관<br>
  우리가 Q-value 가 필요한 것은 결국 advantage function을 구하기 위함인데, TD error는 A^\phi의 unbiased estimation 이므로 q-value를 구하지 않아도 된다.

A2C의 advantage 수식을 critic에 대해서 다양한 Time-Scales를 적용하면 (MC, TD, TD람다[forwrad,backwrad])
- Monte-Carlo
{{< katex display =true >}}
\Delta\theta \;=\; \alpha\,\big( v_t \;-\; V_\theta(s) \big)\,\phi(s)
{{< /katex >}} 
- TD(0)
{{< katex display =true >}}
\Delta\theta \;=\; \alpha\,\big( r \;+\; \gamma V(s') \;-\; V_\theta(s) \big)\,\phi(s)
{{< /katex >}} 
- forwrad-view TD 람다
{{< katex display =true >}}
\Delta\theta \;=\; \alpha\,\big( v_t^{\lambda} \;-\; V_\theta(s) \big)\,\phi(s)
{{< /katex >}} 
- backward-view TD 람다
{{< katex display =true >}}
\delta_t \;=\; r_{t+1} \;+\; \gamma\, V(s_{t+1}) \;-\; V(s_t) \\
e_t \;=\; \gamma \lambda\, e_{t-1} \;+\; \phi(s_t) \\
\Delta\theta \;=\; \alpha\, \delta_t\, e_t \
{{< /katex >}} 

A2C의 advantage 수식을 actor에 대해서 다양한 Time-Scales를 적용하면 (MC, TD, TD람다[forwrad,backwrad])
 - 다음과 같은 기본식에 대해서 추가적으로 time-scales를 적용할 수 있다.
 {{< katex display =true >}}
\nabla_\theta J(\theta)
= \mathbb{E}_{\pi_\theta}\!\left[\,\nabla_\theta \log \pi_\theta(s,a)\; A^{\pi_\theta}(s,a)\,\right]
{{< /katex >}} 
- Monte-Carlo policy gradient 가 complete return error를 쓸 경우 
{{< katex display =true >}}
\Delta \theta \;=\; \alpha\,\big( v_t - V_w(s_t) \big)\, \nabla_\theta \log \pi_\theta(s_t, a_t)
{{< /katex >}} 
- one-step TD error를 쓸경우
{{< katex display =true >}}
\Delta \theta \;=\; \alpha\,\big( r + \gamma V_w(s_{t+1}) - V_w(s_t) \big)\, \nabla_\theta \log \pi_\theta(s_t, a_t)
{{< /katex >}} 

policy gradient에 동일하게 TD 람다 적용하는 것 처럼 Eligibility Trace 할 수 있다.
 - forward이고
{{< katex display =true >}}
\Delta \theta \;=\; \alpha\,\big( v_t^{\lambda} - V_{w}(s_t) \big)\, \nabla_\theta \log \pi_\theta(s_t, a_t)
{{< /katex >}} 
 - forward 와 backward가 동치가 되는 점을 정리하면 다음과 같다.
{{< katex display =true >}}
\delta_t \;=\; r_{t+1} + \gamma\, V_{w}(s_{t+1}) - V_{w}(s_t) \\
e_{t+1} \;=\; \lambda\, e_t + \nabla_\theta \log \pi_\theta(s_t, a_t) \\
\Delta \theta \;=\; \alpha\, \delta_t\, e_t
{{< /katex >}} 
 - 이것으 장점은 episode가 끝나지 않아도 online 업데이트 한다.

 - 강의에서는 natural Actor-Critic을 설명하는데, posting에서는 생략함.
