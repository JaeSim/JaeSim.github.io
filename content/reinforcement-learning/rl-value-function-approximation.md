+++
date = '2025-09-17T18:00:56+09:00'
title = '6. Value Function Approximation'
weight = 9
tags = ["Definition", "Model Free", "Value Function", "approximiation", "Reinforcement Learning", "RL"]
categories = ["Reinforcement Learning"]
+++


# **6. Value Function Approximation**

large MDP를 풀수 없으니 (too many state and action) value function(state-value/action-value function)을 어떻게 근사하게 구하는가?
- Liner combinations of feature
- Neural network



{{< katex display = true >}}

\hat{v}(s, \mathbf{w}) \approx v_\pi(s)
\quad \text{or} \quad
\hat{q}(s, a, \mathbf{w}) \approx q_\pi(s,a)

{{< /katex >}}


여기서부터 Gradient Descent 가 나온다.

>1) 정책 π 고정 
>2) Q-function을 gradient descent로 근사 (policy evaluation) 
>3) 근사된 Q 기반으로 정책 개선 (policy improvement) 
>4) 다시 반복 (⇒ 점진적으로 최적 정책에 수렴)


value function을 근사해서 사용하므로 **value-based** 라고도 한다.

 - IID 트레이닝 method는 RL방법론에 적합하지 않다. (RL은 이전행동이 현재 행동에 크게 영향을 미치므로)

 - neural network처럼 한스탭마다 value function을 online 업데이트하는 incremental method와 한번에 모아서 하는 batch method가 있다(data efficiency). <br>이것은 RL 관점에서는 중요한 분류가 아니지만, RL학습 코스(David Silver)에서는 이를 순차적으로 소개한것 정도로만 집고 넘어가자

 

## **Incremental Method**

### **Value Function Approximation**
아래와 같이 action value function은 approximate를 통해서 표현될수 있고.
델타 W를 작게하므로써 근사를 구할 수 있다.

- **먼저 오라클이 있다고 가정하고** {{< katex display = false >}}v_{\pi}(S){{< /katex >}} 나 {{< katex display = false >}}q_{\pi}(S, A){{< /katex >}} 를 우리에게 주어진다고 하면

- {{< katex display = false >}}v_{\pi}(S){{< /katex >}} 나 {{< katex display = false >}}q_{\pi}(S, A){{< /katex >}}
 와 근사 함수를 적게 하기 위해서 J(w)를 최소화하면 되고 (MSE).
{{< katex display = true >}}
J(\mathbf{w}) = \mathbb{E}_{\pi} \left[ \left( v_{\pi}(S) - \hat{v}(S, \mathbf{w}) \right)^2 \right]
\\
J(\mathbf{w}) = \mathbb{E}_{\pi} \left[ \left( q_{\pi}(S, A) - \hat{q}(S, A, \mathbf{w}) \right)^2 \right]
{{< /katex >}}

 - 이걸 미분으로 표현하면 아래와 같다. {{< katex display = false >}} \alpha {{< /katex >}} 는 step size
{{< katex display = true >}}
- \frac{1}{2}\alpha \nabla_{\mathbf{w}} J(\mathbf{w}) 
= \alpha \mathbb{E}_\pi \left[ \bigl(v_\pi(S) - \hat{v}(S, \mathbf{w})\bigr) \nabla_\mathbf{w} \hat{v}(S, \mathbf{w}) \right]

\\
- \frac{1}{2} \nabla_{\mathbf{w}} J(\mathbf{w}) 
= \left( q_{\pi}(S, A) - \hat{q}(S, A, \mathbf{w}) \right) \nabla_{\mathbf{w}} \hat{q}(S, A, \mathbf{w})
{{< /katex >}}

 - 이것을 stochastic gradient descent는 samples 하므로 다음과 같다. 이것은 전체 gradient를 업데이트 한거랑 같다.
{{< katex display = true >}}
\Delta \mathbf{w} = \alpha \bigl(v_\pi(S) - \hat{v}(S, \mathbf{w})\bigr) \nabla_{\mathbf{w}} \hat{v}(S, \mathbf{w})
\\
\Delta \mathbf{w} = \alpha \left( q_{\pi}(S, A) - \hat{q}(S, A, \mathbf{w}) \right) \nabla_{\mathbf{w}} \hat{q}(S, A, \mathbf{w})
{{< /katex >}}

- feature vector로 표현한것[해당 state일때 그 feature 값만 1이고 나머진 0인 모양]은 table lookup 한거랑 결과적으로 비슷한 직관을 가지고 있다.

#### **Linear Function Approximation**
- linear function approximation 은 보통 feature vector를 써서 weight를 곱한다 (david silver 강의 에서는 neural network로 하는것은 다루지 않는다)
{{< katex display = true >}}
\mathbf{x}(S) =
\begin{pmatrix}
x_1(S) \\
\vdots \\
x_n(S)
\end{pmatrix}
\\
\hat{v}(S,\mathbf{w}) = \mathbf{x}(S)^\top \mathbf{w}
= \sum_{j=1}^{n} x_j(S)\,w_j
{{< /katex >}}
 - 이를 MSE로 표현하고 미분하여 정리하면 다음과 같다
{{< katex display = true >}}
\nabla_{\mathbf{w}} \hat{v}(S,\mathbf{w}) = \mathbf{x}(S) \\
\Delta \mathbf{w} = \alpha \bigl(v_\pi(S) - \hat{v}(S,\mathbf{w})\bigr)\mathbf{x}(S)
{{< /katex >}}



### **Incremental Prediction Algorithms**
 - 위에 설명들은 결국 **오라클이 진짜 value function을 주어졌을때** 의 식들이다
 - 우리는 없으니 이것을 MC, TD(0), TD(람다)[forward,backward]로 표현하면 다음과 같다. <br>(수식은 state-value function에 대해서만 있지만 q-value function에서도 가능)
{{< katex display = true >}}
\Delta \mathbf{w} = \alpha \bigl(G_t - \hat{v}(S_t, \mathbf{w})\bigr) \nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w})
\\
\Delta \mathbf{w} = \alpha \bigl(R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})\bigr) \nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w})
\\
\Delta \mathbf{w} = \alpha \bigl(G_t^{\lambda} - \hat{v}(S_t, \mathbf{w})\bigr) \nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w})
\\
\Delta \mathbf{w} = \alpha \, \delta_t \, E_t
{{< /katex >}} 
    - MC는 {{< katex display = false >}}G_t{{< /katex >}} 가 unbiased 고 noisy sampe을 {{< katex display = false >}}v_\pi(S_t){{< /katex >}} 로 부터 받을 수 있으니 supervised learning 을 적용할 수 있다.
    - MC 는 조금 느리지만 **non-linear value** function approximation을 쓰더라도 **local optimum**에 도달한다는 장점이 있다
    - TD는 true value 펑션의 biased 샘플이다. 여전히 supervised learning을 적용할 수 있다. **linear** TD(0)은 **global optimum**에 도달한다.
    - TD(람다)는 true value 펑션의 biased 샘플이다. 동일하게 supervised learning을 적용할 수 있고. <br>
    forward view linear TD(람다) 와 backward view linear TD(람다) 둘다 계산이 가능하다.


### **Incremental Control Algorithms**

 - q-value를 다음과 같은 근사를 사용해도 그림과 같으 수렴한다.{{< katex display = false >}}\hat{q}(\cdot, \cdot, \mathbf{w}) \approx q_\pi{{< /katex >}}
   - 최종 점으로 very close하게되고, 최종점을 하나의 ball 처럼 오실레이트 되는 구간(볼안) 에서 왔다 갔다 한다고 생각하자
   - q-value 미분 관련식은 위에 언급하였음

<img src="/images/rl-value-approximtion-control.png" alt="convergence approximation" style="width:80%;"/>


 - q-value에 대해서 feature vector를 보면 다음과 같고, 
 {{< katex display = true >}}
\mathbf{x}(S, A) =
\begin{pmatrix}
x_1(S, A) \\
\vdots \\
x_n(S, A)
\end{pmatrix} \\
\hat{q}(S, A, \mathbf{w}) = \mathbf{x}(S, A)^\top \mathbf{w}
= \sum_{j=1}^{n} x_j(S, A) w_j \\
\Delta \mathbf{w} = \alpha \bigl(q_\pi(S, A) - \hat{q}(S, A, \mathbf{w})\bigr) \mathbf{x}(S, A)
{{< /katex >}} 
 - 이것을 state-value 와 같이 MC,TD,TD(람다)[forward,backward]를 적용할 수 있다.
 {{< katex display = true >}}
\Delta \mathbf{w} = \alpha \bigl(G_t - \hat{q}(S_t, A_t, \mathbf{w})\bigr) \nabla_{\mathbf{w}} \hat{q}(S_t, A_t, \mathbf{w})
\\
\Delta \mathbf{w} = \alpha \bigl(R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}) - \hat{q}(S_t, A_t, \mathbf{w})\bigr) \nabla_{\mathbf{w}} \hat{q}(S_t, A_t, \mathbf{w})
\\
\Delta \mathbf{w} = \alpha \bigl(q_t^{\lambda} - \hat{q}(S_t, A_t, \mathbf{w})\bigr) \nabla_{\mathbf{w}} \hat{q}(S_t, A_t, \mathbf{w})
\\
\Delta \mathbf{w} = \alpha \, \delta_t \, E_t
{{< /katex >}} 


### **Convergence of Control Algo**
TD는 bootstrap은 좋은 효과를 가졌으나 수렴하지 않을 수 도 있다.<br>
MC은 on-policy, off-policy에서 수렴하나 TD나 어떤 approximation을 사용하는지 따라 수렴하지 않을 수 도있다. <br>
<img src="/images/rl-mc-td-convergence.png" alt="table mc,td convergence" style="width:80%;"/>

 - off policy일때 TD는 linear function approx 를 써도 수렴하지 않는다.
    - 왜냐하면, Gradient Descent 아님: TD(0) 업데이트는 실제 손실 함수의 기울기 방향이 아니라 Projected Bellman Equation을 근사하는 방향으로 움직입니다. 따라서 "진짜 기울기"를 따라가지 않음 by GPT
 - gardient TD 는 TD의 이러한 문제점들을 수정한 것 
   - Mean-Squared Projected Bellman Error (MSPBE) 최소화
   - 보조 변수 h 추가 (Two-timescale Update)
   - two-timesacle stochastic approximation에 의해서, 전역 수렴성을 보장

<img src="/images/rl-mc-td-convergence_2.png" alt="table mc,td convergence" style="width:80%;"/>

 - method를 정리하면 다음과 같다.

<img src="/images/rl-mc-td-convergence_3.png" alt="table mc,td convergence" style="width:80%;"/>

## **Batch Method**

### **Experience Replay**
exprience 한번을 통해서 기울기를 한번 업데이트하는건 data efficiency 하지 않다.<br>
위의 gradient descent 할때 sampling을 효율적으로 하기 위한 여러가지 방법

아래와 같은 근사함수가 있고.
{{< katex display = true >}}
\hat{v}(s, \mathbf{w}) \approx v_\pi(s)
{{< /katex >}} 

Oracle이 도와줘서 우리는 못구하는  {{< katex display = false >}}v_\pi{{< /katex >}}  를 정확하게 알수 있다고 가정하자.

이것들을 dataset에 가지고 있다고 치자. 전체 dataset은 다음과 같이 표현한다.
 {{< katex display = true >}}
\mathcal{D} = \{ \langle s_1, v_1^\pi \rangle, \langle s_2, v_2^\pi \rangle, \ldots, \langle s_T, v_T^\pi \rangle \}
{{< /katex >}} 


그렇다면 우리가 근사하는 value function이  {{< katex display = false >}}v_\pi{{< /katex >}} 로부터 얼마나 떨어져 있는지 알수 있다.<br>
이것을 이용해서 mean square error를 수식으로 나타낼 수 있고,
이것을 작게 하는것 (=value 근사가 잘되는것) 을 찾아햔다.

least squear 알고리즘으로 w를 최소하하는 수식은 다음과 같다. 이것을 expectation으로 쓸수 있다.
{{< katex display = true >}}
LS(\mathbf{w}) = \sum_{t=1}^T \bigl(v_t^\pi - \hat{v}(s_t, \mathbf{w})\bigr)^2
= \mathbb{E}_{\mathcal{D}}\left[\bigl(v^\pi - \hat{v}(s, \mathbf{w})\bigr)^2\right]
{{< /katex >}} 

빠른/쉬운 방법은, 
기존 값들을 저장하고, 학습할때 랜덤하게 샘플링해서, <br>
stochastic gradient desent update하는 것이고, 이것을 반복한다 (=Experience Replay)<br>
이것은 사실 standard supervised learning 이긴 하다. 

 - D 로부터 랜덤하게 샘플링하고, stochastic gradient desent update 를 한다. 그리고 이것을 repeat한다.
{{< katex display = true >}}
\langle s, v^\pi \rangle \sim \mathcal{D}
\\
\Delta \mathbf{w} = \alpha \bigl(v^\pi - \hat{v}(s, \mathbf{w})\bigr) \nabla_{\mathbf{w}} \hat{v}(s, \mathbf{w})
{{< /katex >}} 
 - 결국 다음과 같이 수렴한다,
{{< katex display = true >}}
\mathbf{w}^\pi = \arg\min_{\mathbf{w}} LS(\mathbf{w})
{{< /katex >}} 

### **Experience Replay in DQN**

DQN 은 experience Replay 와 fixed Q-target 의 두가지 테크닉을 써서 수렴한다.<br>
 - fixed Q-target : 현재 학습중인 네트워크와 타겟을 계산할때 쓰는 네트워크를 분리 (타겟이 너무 자주 바뀌지 않아서 안정적인 학습을 위함)
    - 참고로 TD 러닝은 fixed 파라미터가 아니라서 발산 위험이 있음
 - 수식으로보면 오라클이 알려준다고 가정하는 진실된 value function을 old network가 대체되는셈
    

DQN는 다음과 같다
 - {{< katex display = false >}}\epsilon{{< /katex >}}  greedy policy를 사용해서 action 선택
 - {{< katex display = false >}}(s_t, a_t, r_{t+1}, s_{t+1}){{< /katex >}} 를 {{< katex display = false >}}\mathcal{D}{{< /katex >}} 에 저장
 - sampe random mini-batch를 진행.  {{< katex display = false >}}\mathcal{D}{{< /katex >}} 에 있는 {{< katex display = false >}}(s, a, r, s'){{< /katex >}}를 사용
 - old fixed 파라미터를 이용해서 {{< katex display = false >}}w^-{{< /katex >}}을 이용해서 Q러닝 계산
 - 이때 MSE 수식은 다음과 같음
{{< katex display = true >}}
\mathcal{L}_i(w_i) = \mathbb{E}_{s,a,r,s' \sim \mathcal{D}_i} 
\left[ \left( r + \gamma \max_{a'} Q(s', a'; w_i^-) - Q(s, a; w_i) \right)^2 \right]
{{< /katex >}} 
 - stochastic gradient descent 사용

### **Linear Least Squarses Control**
 - Least square solution을 하는데 이것은 많은 iteration이 필요하니
 - linear value function approximation을 사용해서, least squares solution을 바로 푸는것
   - Sherman–Morrison 을 이용해서 {{< katex display = false >}}O(N^3){{< /katex>}} 에서 {{< katex display = false >}}O(N^2){{< /katex>}} 로 줄인다. 
   - linear이니  {{< katex display = false >}} \hat{v}(s, \mathbf{w}) = \mathbf{x}(s)^\top \mathbf{w}{{< /katex >}} 와 같고 수식적으로 아래와 같다.
{{< katex display = true >}}
\mathbb{E}_{\mathcal{D}}[\Delta \mathbf{w}] = 0 \\
\alpha \sum_{t=1}^T \mathbf{x}(s_t) \bigl(v_t^\pi - \mathbf{x}(s_t)^\top \mathbf{w}\bigr) = 0 \\
\sum_{t=1}^T \mathbf{x}(s_t) v_t^\pi = \sum_{t=1}^T \mathbf{x}(s_t)\mathbf{x}(s_t)^\top \mathbf{w} \\
\mathbf{w} = \left(\sum_{t=1}^T \mathbf{x}(s_t)\mathbf{x}(s_t)^\top \right)^{-1} \sum_{t=1}^T \mathbf{x}(s_t) v_t^\pi
{{< /katex >}} 
 - LSMC, LSTD, LSTD(람다) 에 적용이 가능하다.
{{< katex display = true >}}
\mathbf{w} = \left( \sum_{t=1}^T \mathbf{x}(S_t)\mathbf{x}(S_t)^\top \right)^{-1} \sum_{t=1}^T \mathbf{x}(S_t) G_t \\
\mathbf{w} = \left( \sum_{t=1}^T \mathbf{x}(S_t)\bigl(\mathbf{x}(S_t) - \gamma \mathbf{x}(S_{t+1})\bigr)^\top \right)^{-1} \sum_{t=1}^T \mathbf{x}(S_t) R_{t+1} \\
\mathbf{w} = \left( \sum_{t=1}^T E_t \bigl(\mathbf{x}(S_t) - \gamma \mathbf{x}(S_{t+1})\bigr)^\top \right)^{-1} \sum_{t=1}^T E_t R_{t+1}
{{< /katex >}} 

