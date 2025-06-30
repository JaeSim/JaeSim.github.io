+++
date = '2025-06-23T10:01:40+09:00'
title = 'Learned Scheduler'
subtitle =  'Learned Scheduler에 대한 직관을 얻기 위한 posting'
weight = 6
tags = ["RL", "ML", "NN", "Database"]
categories = ["Learned Scheduler"]
+++

## **Learned Scheduler 관련 paper 리서치**
### **LSched: A Workload-Aware Learned Query Scheduler for Analytical Database Systems(2022)**
- **읽어봐야할 논문**
- SIGMOD22, 28회 인용 MIT 팀.
- a fully RL-based learned workload-aware query scheduler for in-memory analytical database systems.
- https://dl.acm.org/doi/10.1145/3514221.3526158
- LS(Learned Scheduler)를 처음 논문. (RL을 적용했던 Decima와는 학습 대상이 다름)
- **추가적으로 논문 내용에 _Decima 와 다르게_ 라는 언급이 매우 많이 되어 있음**
- Decima가 비슷하게 RL 베이스드 learning이라고 related works에 언급<br>
저자에 따르면, Decima는 cluster간의 스케쥴링이고, LSched는 단일 데이터베이스 내부를 말하지만, 여러가지 비교언급이 되어있음.
> Decima [34], which uses RL to fully-learn a jobs scheduler on large clusters, <br>
> is the closest to LSched in the approach, but different in the scheduling objective. <br>
> Decima aims to schedule tasks among large cluster nodes,
- In-memory 까지만 적용범위
- 최신 Query Encoder 기법을 제시 (with Graph Attention Network)
- 최초의 operation pipeline을 한 논문이라고 언급
> As far as we know, automatically controlling the operators pipelining was never introduced before by previous schedulers including Decima.
- Decima 는 black-box feature(number of tasks)를 사용, LsChed는 white-box feature를 사용(fine-gained level work orders)
- by GPT

| 구분           | Decima(Black-box)                        | LSched(White-box)                                |
|----------------|-----------------------------------------|------------------------------------------------|
| 관점           | 내부 구조를 숨김                        | 내부 구조를 노출                               |
| 접근 방식      | 단순 상태(완료 여부)만 관찰            | 연산자 세부정보와 관계까지 고려               |
| 파이프라이닝   | 불가능 (부모 완료 후 자식 실행)        | 가능 (_**동시에 실행 가능성 고려**_)                |
| 최적화 가능성  | 제한적                                  | 세밀한 최적화 가능                             |
 - 느낀점. Decima 에서 많이 대비하여 설명

## **Learning scheduling algorithms for data processing clusters(2018)**
- SIGMOD19, 870회 인용, MIT와 칭화대
- = Decima 
- https://dl.acm.org/doi/10.1145/3341302.3342080
- https://www.themoonlight.io/ko/review/learning-interpretable-scheduling-algorithms-for-data-processing-clusters
- https://ita9naiwa.github.io/reinforcement%20learning/2019/02/20/scheduling-dpc.html