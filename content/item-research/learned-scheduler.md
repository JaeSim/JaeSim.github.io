+++
date = '2025-07-03T10:53:40+09:00'
title = 'Learned Scheduler'
subtitle =  'Learned Scheduler에 대한 직관을 얻기 위한 posting'
weight = 6
tags = ["RL", "ML", "NN", "Database"]
categories = ["Learned Scheduler"]
+++

## **Learned Scheduler 관련 paper 리서치**
### **LSched: A Workload-Aware Learned Query Scheduler for Analytical Database Systems(2022)**
- SIGMOD22, 28회 인용 MIT 팀. (***Non-open source***)
- a fully RL-based learned workload-aware query scheduler for ***in-memory analytical database systems***.
- https://dl.acm.org/doi/10.1145/3514221.3526158
- LS(Learned Scheduler)를 처음 언급. (RL을 적용했던 Decima와는 학습 대상이 다름)
- **추가적으로 논문 내용에 _Decima 와 다르게_ 라는 언급이 매우 많이 되어 있음**
- ***Decima 가 database-specific characteristics를 고려하지 않은 한계점을 언급 (query structure, operator)<br>
   이로인한 pipelining 기회를 날린다고 주장.***
- Decima 와의 비교
  - Decima가 비슷하게 RL baseed learning이라고 related works에 언급<br>
  저자에 따르면, Decima는 cluster간의 스케쥴링이고, LSched는 단일 데이터베이스 내부를 말하지만, 여러가지 비교언급이 되어있음.
      > Decima [34], which uses RL to fully-learn a jobs scheduler on large clusters, <br>
      > is the closest to LSched in the approach, but different in the scheduling objective. <br>
      > Decima aims to schedule tasks among large cluster nodes,
  - In-memory 까지만 적용범위
  - 최신 Query Encoder 기법을 제시 (with Graph Attention Network)
  - 최초의 operation pipeline을 한 논문이라고 언급
      > As far as we know, automatically controlling the operators pipelining was never introduced before by previous schedulers including Decima.
  - Decima 는 black-box feature(number of tasks)를 사용, LSched는 white-box feature를 사용(fine-gained level work orders)
  - by GPT

  | 구분           | Decima(Black-box)                        | LSched(White-box)                                |
  |----------------|-----------------------------------------|------------------------------------------------|
  | 관점           | 내부 구조를 숨김                        | 내부 구조를 노출                               |
  | 접근 방식      | 단순 상태(완료 여부)만 관찰            | 연산자 세부정보와 관계까지 고려               |
  | 파이프라이닝   | 불가능 (부모 완료 후 자식 실행)        | 가능 (_**동시에 실행 가능성 고려**_)                |
  | 최적화 가능성  | 제한적                                  | 세밀한 최적화 가능                             |
 - 느낀점. <br>
 Decima 에서 많이 대비하여 설명. <br>
 후에 나온 BQSched(2025)는 오히려 다시 Black-box가 좋다는식으로 나와서... 흠;

### **Learning scheduling algorithms for data processing clusters(2019)**
- SIGMOD19, 870회 인용, MIT와 칭화대. Project 명 : **Decima**
- https://dl.acm.org/doi/10.1145/3341302.3342080
- https://www.themoonlight.io/ko/review/learning-interpretable-scheduling-algorithms-for-data-processing-clusters
- https://ita9naiwa.github.io/reinforcement%20learning/2019/02/20/scheduling-dpc.html
- https://github.com/hongzimao/decima-sim
- Spark 에서 실험했으며, 한정된 자원속에서 DAG문제를 RL로 푸는 방법을 제시
- Query Job에 대한 새로운 representation 을 제시
  - per-node embbeding, per-job and global embbeding
- 여러가지 RL 테크닉 접목 ("streaming scenario" 에서 오는 problem 우회)
  -  curriculum learning 기반 초기화 전략 & random termination
       - Batch 시나리오로 학습시 streaming 에서 성능이 떨어지도록 학습됨 <br>
       streaming일경우 queue에 작업이 과도하게 쌓이니 , 이를 방지하기 위해서 초반에는 early terminate함. <br>
       이후에는 더 긴 시퀀스 및 어려운 문제를 해결하도록 유도 (2009. Curriculum learning)
       - **이경우 언제 종료될지 학습할수 있어 항상 종료시간직전까지 미루는 전략으로 학습될 수 있음**
       - memoryless 종료 방식을 사용해서, 지수 분포에서 무작위 샘플링으로 특정 시간이후 종료되도록 하고, <br>
      실험이 지속될수록 이 평균시간을 늘려나감
  - variance reduction technique 접목 (2019. Variance Reduction for Reinforcement Learning in Input-Driven Environments.)
    - 각 작업의 도작(task 수행의 종료시점)의 randomness에 의해서, (행하는 task자체가 원래 짧게 걸리는 쿼리거나, 원래 수행시간이 오래걸리는 쿼리거나), 보상이 무작위함
    - 각 도착마다 baseline을 별도로 계산하여 정확하게 평가
 - 느낀점<br>
 실시간 online RL 에 대해서는? 유기적으로 동작하는 것을 만드는건 어떠한가?

### **BQSched: A Non-intrusive Scheduler for Batch Concurrent Queries via Reinforcement Learning(2025)**
- ICDE2025, 
- RL 기반 query job scheduler
- 기존 방식들 (LSched 등) 은 instrusive (침습적?) 이라고 말함. (DBMS와 통합해야한다.) 따라서 DBMS가 blackbox라 가정하고, 할수 있는 방법으로 제시
- dbms 에 전달하기 전에 별도의 scheduler를 만들어서 돌리고, 이로인한 결과를 DBMS에 plan된 batch 순서에 맞게 전달
- attention + PPO 기반 scheduler
