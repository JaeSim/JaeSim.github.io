+++
date = '2025-07-31T09:45:54+09:00'
title = '2025 SIGMOD'
weight = 11
tags = ["Conferenece", "Paper", "Research", "Survey", "Database", "SIGMOD"]
categories = ["Reserach", "Study"]
+++

# **SIGMOD 2025 Research Track**
 - https://2025.sigmod.org/sigmod_papers.shtml
 - 논문이 워낙 많기 때문에 ***직접 읽지 않고, GPT 4o, gemini 2.5-pro*** 를 통해서 요약한 것을 posting한것
 - 최대한 거르긴 했는데,, ***잘못 오기입 되어 있을 수 있으므로, 필히 직접 확인 필요***
 
## **Summary**
### **느낀점**
 - 데이터베이스, 빅데이터, 빅데이터를 위한 시스템, 인덱스와 같은 알고리즘적 증명 등이 포함되어있고, 그래프 시큐리티에 대해서 꽤 비중이 있음
 - ICDE, EDBT에 유사한 내용이 있는것 같기도..
 - AI 요약을 하나한 abstact에 넣어주지 않으면 헛소리를 하기 때문에, 직접 검수가 필요함

### **Interested Papers**
 - AJOSC: Adaptive join order selection for continuous queries on data streams
   - Continuous Queries, Multi-way Join, Join Order Selection, Adaptive Algorithm, Data Streams
   - 기존 연속 다중-방향 조인 순서 선택 방식은 휴리스틱에 의존하거나 재계산 비용이 높아 비효율적인 문제가 있음.
   - 새로운 비용 모델과 증분 재최적화 알고리즘을 결합하여 데이터 변화에 적응하며 최적의 조인 순서를 효율적으로 찾아 해결함.
   - **흥미로운 or 신기한**
   
 - How Good are Learned Cost Models, Really? Insights from Query Optimization Tasks
   - Learned Cost Models (LCMs), Query Optimization, Cost Estimation, Join Ordering, Access Path Selection, Physical Operator Selection
   - 학습된 비용 모델(LCM)이 실제 쿼리 최적화 작업에서 얼마나 효과적인지에 대한 검증이 부족한 문제가 있음.
   - 핵심 쿼리 최적화 작업에서 최신 LCM과 전통적 비용 모델을 비교 평가하여 실효성을 검증하고 향후 연구 방향을 제시함.
   - **흥미로운 or 신기한**
   
 - Intra-Query Runtime Elasticity for Cloud-Native Data Analysis
   - Intra-Query Runtime Elasticity (IQRE), Cloud-Native Data Analysis, Degree of Parallelism (DOP), Query Engine, Elasticity
   - 기존 클라우드 분석 엔진은 쿼리 실행 중 병렬 처리 수준을 동적으로 조정할 수 없어 자원 사용이 비효율적인 문제가 있음.
   - 데이터 처리 중단 없이 쿼리 실행 중 병렬 처리 수준을 동적으로 조절하여 비용 효율성을 높이는 Accordion 엔진을 제시함.
   - **흥미로운 or 신기한**
   
 - Debunking the Myth of Join Ordering: Toward Robust SQL Analytics
   - Join Order Optimization, Robust Query Processing, Predicate Transfer, Acyclic Queries, DuckDB
   - 기존 최적화기는 조인 순서에 민감해 최적보다 매우 느린 계획을 생성할 수 있는 문제가 있음.
   - RPT 알고리즘을 제안하여 임의 조인 순서에 견고하고 빠른 실행 성능을 제공함으로써 이 문제를 해결함.
   - **흥미로운 or 신기한**
   
 - Incremental Rule Discovery in Response to Parameter Updates
   - Rule Mining, Support & Confidence Thresholds, Incremental Algorithms, Entity Enhancing Rules, Parameter Tuning
   - 임계값 변경 시 매번 규칙을 처음부터 재탐색해야 하므로 계산 비용이 큰 문제가 있음.
   - 변경된 부분만 계산하는 최적 증분 알고리즘을 통해 이 문제를 해결함.
   - **흥미로운 or 신기한**
   
 - Athena: An Effective Learning-based Framework for Query Optimizer Performance Improvement
   - Query Optimization, Learned Models, Execution Plan Exploration, PostgreSQL, Cost Models
   - 기존 학습 기반 최적화기는 계획 탐색 공간이 제한되고 실행 계획으로부터 비효율적으로 학습되는 문제가 있음.
   - 순서 중심 탐색기와 Tree-Mamba 비교기 및 시간 가중 손실 함수를 통해 학습 효율을 개선하여 이 문제를 해결함.
   - **흥미로운 or 신기한**
   
 - RWalks: Random Walks as Attribute Diffusers for Filtered Vector Search
   - Filtered Vector Search, Attribute Filtering, Graph-based Index, Random Walks, HNSW
   - 필터가 포함된 벡터 검색 시, 결과를 모두 찾은 후 필터링하는 방식은 필터의 선택도가 높을 때 매우 비효율적인 문제를 가지고 있음.
   - 랜덤 워크를 통해 벡터 유사도 그래프에 속성 정보를 미리 '확산'시켜, 검색 시 필터를 만족할 만한 후보로 탐색 경로를 유도하는 방식을 제안함.
   - **흥미로운 or 신기한**

 - Learned Offline Query Planning via Bayesian Optimization
   - Query Optimization, Learned Query Planning, Bayesian Optimization, Offline Learning, Black-box Optimization
   - 전통적인 쿼리 옵티마이저는 부정확한 비용 모델로 인해 최적의 실행 계획을 찾지 못하는 경우가 많고, 학습 기반 옵티마이저는 온라인 학습으로 인한 부담이 있음.
   - 실제 쿼리 실행 시간을 블랙박스 함수로 보고, 베이지안 최적화를 사용해 오프라인 환경에서 직접 쿼리를 실행하며 최적의 실행 계획을 학습하는 방식을 제안함.
   - **흥미로운 or 신기한**
   
 - Low Rank Learning for Offline Query Optimization
   - Query Optimization, Cardinality Estimation, Learned Optimizer, Low-Rank Learning, Matrix Completion, Offline Learning
   - 학습 기반 쿼리 옵티마이저는 모델이 복잡하고 파라미터가 많아, 학습에 많은 데이터와 시간이 필요하며 과적합되기 쉬운 문제를 가지고 있음.
   - 쿼리들의 카디널리티 관계를 '저차원 행렬'로 간주하고, 행렬 완성(matrix completion) 기법을 통해 적은 데이터로도 효율적으로 학습하는 방식을 제안함.
   - **흥미로운 or 신기한**
   
 - Rapid Data Ingestion through DB-OS Co-design
   - Data Ingestion, Sequential Access, DB-OS Integration, I/O Optimization, zicIO
   - 기존 시스템은 순차적 적재 시 계층 간 지연과 중복 데이터 접근으로 인한 성능 저하 문제가 있음
   - DB와 OS 간 협업으로 데이터 접근을 자동화하고 중복 페칭을 제거하여 고속 적재 문제를 해결함.
   - **흥미로운 or 신기한**

 - Efficiently Processing Joins and Grouped Aggregations on GPUs
   - GPU Databases, Joins, Group-By, Random Access, Query Optimization
   - 기존 GPU 기반 연산은 무작위 접근과 group-by 처리 부족으로 성능 저하 문제가 있음.
   - GFTR 기법과 최적화된 group-by 알고리즘을 통해 성능 병목을 해결하고 효율적으로 처리함.
   - **흥미로운 or 신기한**
   
 - LeaFi: Data Series Indexes on Steroids with Learned Filters
   - Data Series, Similarity Search, Learned Index, Pruning Optimization
   - 기존 시계열 인덱스는 가지치기 효율이 낮아 검색 연산 낭비가 큰 문제가 있음.
   - 노드 간 거리 하한을 예측하는 학습 기반 필터를 통해 가지치기를 고도화하여 검색 속도를 향상시킴.
   - **흥미로운 or 신기한**

 - An Elephant Under the Microscope: Analyzing the Interaction of Optimizer Components in PostgreSQL
   - Query Optimizer, PostgreSQL, Cardinality Estimation, Cost Model, Plan Generation 
   - 질의 최적화기의 구성 요소 간 상호작용이 충분히 이해되지 않아 예상 외의 결과나 연구 낭비 문제가 있음.
   - PostgreSQL 기반 실험 분석을 통해 구성 요소 간 영향을 규명하고, 개선 방향을 제시함.
   - **흥미로운 or 신기한**

 - DPconv: Super-Polynomially Faster Join Ordering
   - Join Ordering, Query Optimization, Subset Convolution, Dynamic Programming, Cost Functions
   - 기존 조인 순서 결정 알고리즘은 O(3ⁿ) 복잡도로 대규모 쿼리에서 실행 시간이 과도한 문제가 있음.
   - 부분집합 컨볼루션 기반의 DPconv 프레임워크로 해당 복잡도를 극복하여 실행 시간을 대폭 줄이는 방식으로 해결함.
   - **흥미로운 or 신기한**
   
 - Live Patching for Distributed In-Memory Key-Value Stores
   - Live Patching, Redis Cluster, Rolling Update, High Availability, In-Memory Database
   - 기존 롤링 업데이트는 메모리 상태 복원·동기화로 인해 보안 패치가 지연되는 문제가 있음.
   - 노드 재시작 없이 메모리 내 직접 패치하는 경량 라이브 패칭으로 문제를 해결함.
   - **흥미로운 or 신기한**


## **Detail**
 - PilotDB: Database-Agnostic Online Approximate Query Processing with A Priori Error Guarantees
   - Approximate Query Processing (AQP), Error Guarantees, Database-Agnostic, Block-level Sampling
   - 기존 근사 질의 처리 기술은 사용자 지정 오류율 보장, 유지보수 오버헤드 제거, DBMS 수정 회피를 동시에 달성하기 어려웠음.
   - TAQA 및 BSAP 알고리즘을 미들웨어에 구현하여 오류율 보장, 유지보수 제거, DBMS 수정 회피라는 세 가지 목표를 모두 달성함.

 - Low-Latency Transaction Scheduling via Userspace Interrupts: Why Wait or Yield When You Can Preempt?
   - Preemptive Scheduling, Userspace Interrupts, Low-Latency, Transaction Scheduling
   - 기존의 비선점형/협력적 스케줄링은 긴 트랜잭션이 CPU를 독점하여 짧은 고우선순위 트랜잭션의 지연을 유발하는 문제가 있음.
   - 최신 CPU의 유저 공간 인터럽트를 활용한 선점형 스케줄링 엔진(PreemptDB)을 구현하여 이 문제를 해결함.

 - Fast Maximum Common Subgraph Search: A Redundancy-Reduced Backtracking Approach
   - Maximum Common Subgraph (MCS), Backtracking, Graph Similarity, Redundancy Reduction
   - 기존 최대 공통 부분 그래프 탐색 알고리즘은 이론적 보장이 없거나 실제 실행 속도가 느린 문제가 있음.
   - 중복 계산을 줄이는 새로운 백트래킹 알고리즘(RRSplit)을 제안하여 속도와 이론적 보장을 모두 달성함.

 - RM^2: Answer Counting Queries Efficiently under Shuffle Differential Privacy
   - Shuffle Differential Privacy, Matrix Mechanism, Message Complexity, Answer Counting Queries
   - 셔플-DP 모델에서 매트릭스 메커니즘을 순진하게 적용하면 사용자당 메시지 수가 많아 비효율적인 문제가 있음.
   - 메시지 수를 줄이면서 중앙-DP 수준의 정확도를 유지하는 개선된 메커니즘을 제안하여 효율성을 높임.

 - Hydro: Adaptive Query Processing of ML Queries
   - Adaptive Query Processing (AQP), ML Queries, User-Defined Functions (UDFs), Query Optimization
   - ML 질의는 UDF 통계 예측이 어렵고 데이터에 따라 최적 계획이 변해 정적 최적화 방식으로는 비효율적인 문제가 있음.
   - 적응형 질의 처리(AQP)를 통해 UDF 통계를 지속적으로 감시하고 술어 평가 순서와 자원을 동적으로 최적화하여 해결함.

 - Alsatian: Optimizing Model Search for Deep Transfer Learning
   - Transfer Learning, Model Search, Deep Learning, Optimization, Caching
   - 딥러닝 전이 학습 시 수많은 후보 모델을 각각 평가하는 모델 탐색 과정의 높은 계산 비용이 병목 현상을 유발하는 문제가 있음.
   - 모델 간 중복을 활용하여 블록 단위 공유, 중간 결과 캐싱, 탐색 순서 최적화 기법으로 이 문제를 해결함.

 - AJOSC: Adaptive join order selection for continuous queries on data streams
   - Continuous Queries, Multi-way Join, Join Order Selection, Adaptive Algorithm, Data Streams
   - 기존 연속 다중-방향 조인 순서 선택 방식은 휴리스틱에 의존하거나 재계산 비용이 높아 비효율적인 문제가 있음.
   - 새로운 비용 모델과 증분 재최적화 알고리즘을 결합하여 데이터 변화에 적응하며 최적의 조인 순서를 효율적으로 찾아 해결함.
   - **흥미로운 or 신기한**

 - Using Process Calculus for Optimizing Data and Computation Sharing in Complex Stateful Parallel Computations
   - Process Calculus, Stateful Parallel Computations, Data Sharing, Computation Sharing, Behavioral Equations
   - 복잡한 상태 기반 병렬 계산에서 데이터 및 계산 공유를 통한 성능 최적화가 어려운 문제가 있음.
   - 프로세스 계산법 기반의 '행동 방정식'을 통해 병렬 프로그램을 변환하고 자동으로 최적화하여 이 문제를 해결함.

 - Self-Enhancing Video Data Management System for Compositional Events with Large Language Models
   - Video Data Management, Large Language Models (LLMs), User-Defined Functions (UDFs), Self-Enhancing System
   - 기존 비디오 시스템은 복잡한 질의를 처리할 때 사전에 정의된 모듈이 필요하여 새로운 하위 작업을 처리하지 못하는 문제가 있음.
   - 대규모 언어 모델(LLM)을 활용해 필요한 모듈을 사용자 정의 함수(UDF)로 자동 생성하여 시스템 스스로 기능을 확장하며 해결함.

 - SWASH: A Flexible Communication Framework with Sliding Window-Based Cache Sharing for Scalable DGNN Training
   - Dynamic Graph Neural Networks (DGNN), Distributed Training, Sliding Window, Communication Framework, Cache Sharing
   - 기존 분산 프레임워크들은 슬라이딩 윈도우 기반 DGNN 훈련 시 스냅샷 및 윈도우 간 통신과 캐시 최적화가 비효율적인 문제가 있음.
   - 유연한 통신 프레임워크와 새로운 파티셔닝 및 슬라이딩 윈도우 기반 캐시 공유 기술을 결합하여 이 문제를 해결함.

 - HoneyComb: A Parallel Worst-Case Optimal Join on Multicores
   - Worst-Case Optimal Join (WCOJ), Parallel Join, Multicore, Shared Memory, Work Skew
   - 기존 병렬 WCOJ 알고리즘은 최상위 변수만 분할하여 작업 왜곡, 스레드 경합, 중복 계산을 유발하는 문제가 있음.
   - 모든 변수를 분할하고, 새로운 인덱스(CoCo)와 재작성 기법을 도입하여 경합과 중복 계산 문제를 해결함.

 - SBSC: A fast Self-tuned Bipartite proximity graph-based Spectral Clustering
   - Spectral Clustering, Bipartite Graph, Self-tuned, Parameter-free, Computational Cost
   - 기존 스펙트럴 클러스터링은 계산 비용이 높고 성능이 외부 파라미터에 의존하는 문제가 있음.
   - 파라미터 없는 이분 그래프와 지역성 기반 희소화 기법으로 계산 비용을 줄이고 품질을 자동으로 향상시키며 해결함.

 - Subgroup Discovery with Small and Alternative Feature Sets
   - Subgroup Discovery, Interpretability, Constraint-based Mining, Alternative Descriptions
   - 기존 부분 그룹 발견 방법은 설명이 복잡하거나 유일하지 않아 해석이 어려운 문제가 있음.
   - 특징 수를 제한하고 다른 특징을 사용하는 대안적 설명을 찾는 제약을 도입하여 이 문제를 해결함.

 - SwiftSpatial: Spatial Joins on Modern Hardware
   - Spatial Joins, Hardware Acceleration, FPGA, Energy Efficiency, Parallel Processing
   - 기존 공간 조인 연산은 병렬/분산 시스템에서도 비용이 많이 들고 시간이 오래 걸리는 문제가 있음.
   - 혁신적인 병렬 처리 및 메모리 관리를 갖춘 FPGA 기반 가속기(SwiftSpatial)를 제안하여 이 문제를 해결함.

 - GTX: A Write-Optimized Latch-free Graph Data System with Transactional Support
   - Graph Data System, Write-Optimized, Latch-free, ACID Transactions, Lock Contention
   - 기존 그래프 시스템은 동일 정점/간선에 대한 동시 업데이트 시 잠금 경합으로 인해 쓰기 성능이 저하되는 문제가 있음.
   - 래치-프리 저장소와 적응형 델타-체인 락킹 프로토콜을 도입하여 잠금 경합을 제거하고 이 문제를 해결함.

 - Malleus: Straggler-Resilient Hybrid Parallel Training of Large-scale Models via Malleable Data and Model Parallelization
   - Hybrid Parallel Training, Large-scale Models, Straggler Resilience, Adaptive Parallelization
   - 기존 하이브리드 병렬 훈련은 일부 장치가 느려지는 동적 낙오자(straggler) 현상에 민감하여 효율이 저하되는 문제가 있음.
   - 낙오자를 GPU별로 감지하고 데이터/모델 병렬화를 동적으로 재계획하여 이 문제를 해결함.

 - Physical Visualization Design: Decoupling Interface and System Design
   - Physical Visualization Design (PVD), Decoupling, Data Interface, System Design, Latency
   - 인터페이스와 시스템 설계가 결합되어 있어, 원하는 지연 시간과 리소스 제약을 만족하는 데이터 인터페이스 구축이 어려운 문제가 있음.
   - 인터페이스와 시스템 설계를 분리하고 주어진 제약에 맞는 최적의 미들웨어 아키텍처를 자동으로 제안하는 도구(Jade)로 해결함.

 - High-Throughput Ingestion for Video Warehouse: Comprehensive Configuration and Effective Exploration
   - Video-ETL (V-ETL), High-Throughput Ingestion, Configuration Space, Reinforcement Learning
   - 기존 비디오 수집(V-ETL) 방식은 구성 공간이 제한적이고 탐색이 비효율적이어서 최적의 수집 계획을 찾기 어려운 문제가 있음.
   - 포괄적인 구성 공간을 정의하고 강화학습 기반 탐색 전략으로 최적의 구성을 찾아 이 문제를 해결함.

 - Revisiting Graph Analytics Benchmarks
   - Graph Analytics, Benchmark, Data Generation, API Usability, Large Language Model (LLM)
   - 기존 그래프 분석 벤치마크는 알고리즘, 데이터, API 사용성 평가 측면에서 한계가 있어 플랫폼 성능을 제대로 평가하지 못하는 문제가 있음.
   - 핵심 알고리즘, 데이터 생성기, LLM 기반 API 사용성 평가 프레임워크를 포함하는 새로운 종합 벤치마크를 제안하여 해결함.

 - Parallel $k$-Core Decomposition: Theory and Practice
   - k-Core Decomposition, Parallel Algorithm, Graph Analysis, Work-efficiency
   - 기존 병렬 k-코어 분해 알고리즘은 작업 효율성과 높은 병렬성을 동시에 달성하기 어려운 문제가 있음.
   - 샘플링, 수직적 세분성 제어(VGC), 계층적 버킷 구조를 결합한 프레임워크로 이 문제를 해결함.

 - Synthesizing Third Normal Form Schemata that Minimize Integrity Maintenance and Update Overheads
   - Third Normal Form (3NF), Schema Design, Dependency Preservation, Update Overhead, Integrity Maintenance
   - 기존 3NF 스키마 설계는 데이터 중복성과 무결성 유지에 드는 업데이트 오버헤드를 고려하지 않는 문제가 있음.
   - 최소 키와 함수 종속성의 수를 파라미터로 사용하여 스키마 설계 시점부터 오버헤드를 최적화하며 해결함.

 - Styx: Transactional Stateful Functions on Streaming Dataflows
   - Stateful Functions-as-a-Service (SFaaS), Transactional Guarantees, Dataflow, Low Latency
   - 기존 SFaaS 접근 방식은 트랜잭션 보장이 약하거나 비효율적인 프로토콜로 인해 지연 시간이 긴 문제가 있음.
   - 데이터플로우 기반 런타임(Styx)과 캐싱 및 조기 커밋 기법을 통해 이 문제를 해결함.

 - Faster and Efficient Density Decomposition via Proportional Response with Exponential Momentum
   - Density Decomposition, Graph Mining, Proportional Response, Exponential Momentum, Fisher Market
   - 기존 밀도 분해 알고리즘들은 대규모 실제 그래프에서 계산적으로 비싸고 느린 문제가 있음.
   - 시장 동역학 모델과 새로운 지수적 모멘텀을 결합한 비례적 응답 알고리즘으로 이 문제를 해결함.

 - Efficient and Accurate Differentially Private Cardinality Continual Releases
   - Cardinality Estimation, Differential Privacy, Continual Release, Data Streams, Memory Efficiency
   - 기존 프라이빗 카디널리티 연속 공개 방법은 메모리 사용량이 너무 많아 실제 적용이 어려운 문제가 있음
   - 효율적인 추정기와 프라이버시 메커니즘을 결합한 새 프레임워크(FC)로 메모리 사용량과 정확도를 모두 개선하며 해결함.

 - Robust Privacy-Preserving Triangle Counting under Edge Local Differential Privacy
   - Triangle Counting, Edge Local Differential Privacy (edge LDP), Graph Privacy, Data Utility
   - 기존 엣지 LDP 기반 삼각형 수 계산은 제한된 데이터 활용과 부정확한 노이즈 제어로 인해 정확도가 낮은 문제가 있음.
   - 더 많은 노이즈 데이터를 활용하는 정점 중심 접근법과 정교한 민감도 분석 및 프라이버시 예산 최적화를 통해 이 문제를 해결함.

 - LICS: Towards Theory-Informed Effective Visual Abstraction of Property Graph Schemas
   - Property Graph Schema, Visual Abstraction, Usability, Schema Comprehension, Labeled Iconized Composite Schema (łics)
   - 기존 속성 그래프 스키마의 시각적 표현은 복잡하고 기능이 제한되어 비전문가가 이해하기 어려운 문제가 있음.
   - HCI, 인지 심리학 이론에 기반한 새로운 시각적 추상화(łics)와 인터페이스(PASCAL)를 제안하여 이 문제를 해결함.

 - Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search
   - Approximate Nearest Neighbor (ANN), Quantization, High-dimensional Vectors, Asymptotic Optimality
   - 최신 양자화 기법(RaBitQ)은 높은 압축률에만 특화되어, 더 많은 메모리를 사용해 정확도를 높이는 것을 지원하지 못하는 문제가 있음.
   - RaBitQ를 확장하고 점근적 최적성을 달성하여, 다양한 메모리 조건에서 정확도와 효율성을 모두 높이는 새로운 양자화 기법으로 이 문제를 해결함.

 - Finding Logic Bugs in Graph-processing Systems via Graph-cutting
   - Graph-processing Systems, Logic Bugs, Software Testing, Graph-cutting
   - 그래프 처리 시스템 전반에 적용할 수 있는 보편적인 논리 버그 탐지 기법이 부재한 문제가 있음.
   - 특정 패턴을 보존하며 그래프를 분할하고, 원본과 부분 그래프의 결과 관계를 비교하는 Graph-cutting 기법으로 이 문제를 해결함.

 - Scalable Complex Event Processing on Video Streams
   - Complex Event Processing (CEP), Video Streams, Deep Video Analytics, Throughput
   - 기존 비디오 분석 시스템은 단순한 질의만 지원하여 시간적 패턴을 감지하는 복잡 이벤트 처리가 어려운 문제가 있음.
   - 복잡한 이벤트 질의를 효율적으로 지원하도록 설계된 새로운 비디오 스트림 처리 시스템(Bobsled)으로 이 문제를 해결함.

 - How Good are Learned Cost Models, Really? Insights from Query Optimization Tasks
   - Learned Cost Models (LCMs), Query Optimization, Cost Estimation, Join Ordering, Access Path Selection, Physical Operator Selection
   - 학습된 비용 모델(LCM)이 실제 쿼리 최적화 작업에서 얼마나 효과적인지에 대한 검증이 부족한 문제가 있음.
   - 핵심 쿼리 최적화 작업에서 최신 LCM과 전통적 비용 모델을 비교 평가하여 실효성을 검증하고 향후 연구 방향을 제시함.
   - **흥미로운 or 신기한**

 - Wait and See: A Delayed Transactions Partitioning Approach in Deterministic Database Systems for Better Performance
   - Deterministic Databases, Transaction Partitioning, Batch Processing, Cross-Partition Operations, Global Optimization
   - 기존 트랜잭션 파티셔닝 방식은 개별 최적화로 인해 배치 내 트랜잭션 간 공통점을 활용하지 못하는 문제가 있음.
   - 트랜잭션 실행을 지연시켜 배치 내 관계를 분석 후 파티셔닝함으로써 전역 실행을 최적화하는 DelayPart 엔진을 제시함.

 - Intra-Query Runtime Elasticity for Cloud-Native Data Analysis
   - Intra-Query Runtime Elasticity (IQRE), Cloud-Native Data Analysis, Degree of Parallelism (DOP), Query Engine, Elasticity
   - 기존 클라우드 분석 엔진은 쿼리 실행 중 병렬 처리 수준을 동적으로 조정할 수 없어 자원 사용이 비효율적인 문제가 있음.
   - 데이터 처리 중단 없이 쿼리 실행 중 병렬 처리 수준을 동적으로 조절하여 비용 효율성을 높이는 Accordion 엔진을 제시함.
   - **흥미로운 or 신기한**

 - Yannakakis+: Practical Acyclic Query Evaluation with Theoretical Guarantees
   - Acyclic Conjunctive Queries, Yannakakis Algorithm, SQL Engines, Query Optimization
   - 기존 Yannakakis 알고리즘은 높은 이론적 보장에도 불구하고 숨겨진 상수 요인으로 인해 실제 시스템에서 사용이 어려운 문제가 있음.
   - Yannakakis 알고리즘을 개선하여 효율성을 높이고 표준 SQL 엔진에 통합 가능한 Yannakakis+로 이 문제를 해결함.

 - FAAQP: Fast and Accurate Approximate Query Processing based on Bitmap-augmented Sum-Product Network
   - Approximate Query Processing (AQP), Sum-Product Network (SPN), Bitmap, Learned Model
   - 기존 AQP 방식은 샘플량 증가나 모델 정확도 부족으로 질의 정확도와 지연 시간을 동시에 만족시키기 어려운 문제가 있음.
   - 비트맵을 결합한 합곱망(BSPN)을 사용하고 저장 공간 예산을 고려한 모델 구성과 비트맵 병합 전략을 통해 이 문제를 해결함.

 - Moving on From Group Commit: Autonomous Commit Enables High Throughput and Low Latency on NVMe SSDs
   - Commit Protocol, NVMe SSD, Low Latency, High Throughput, Logging, Parallelism
   - 기존 커밋 프로토콜은 NVMe SSD의 성능을 제대로 활용하지 못해 높은 처리량과 낮은 지연을 동시에 달성하기 어려운 문제가 있음.
   - SSD 병렬성과 빠른 쓰기를 활용하고 acknowledgment 절차를 병렬화한 Autonomous Commit으로 이 문제를 해결함.

 - Integral Densest Subgraph Search on Directed Graphs
   - Densest Subgraph, Directed Graph, Network Flow, Approximation Algorithm, Graph Mining
   - 기존 DS 알고리즘은 계산 비용이 높거나 근사 품질이 낮아 실용적인 적용이 어려운 문제가 있음.
   - DS와 유사한 새로운 IDS 모델을 정의하고 효율적인 정확 및 근사 알고리즘을 설계하여 이 문제를 해결함.

 - Debunking the Myth of Join Ordering: Toward Robust SQL Analytics
   - Join Order Optimization, Robust Query Processing, Predicate Transfer, Acyclic Queries, DuckDB
   - 기존 최적화기는 조인 순서에 민감해 최적보다 매우 느린 계획을 생성할 수 있는 문제가 있음.
   - RPT 알고리즘을 제안하여 임의 조인 순서에 견고하고 빠른 실행 성능을 제공함으로써 이 문제를 해결함.
   - **흥미로운 or 신기한**

 - Femur: A Flexible Framework for Fast and Secure Querying from Public Key-Value Store
   - Private Information Retrieval (PIR), Key-Value Store, Privacy-Performance Trade-off, Distance-Based Indistinguishability, Learned Index
   - 기존 PIR 방식은 완전한 보안을 제공하지만 성능과 확장성 한계로 인해 대규모 적용이 어려운 문제가 있음.
   - 거리 기반 이론과 러닝 인덱스를 활용해 보안과 성능을 유연하게 절충할 수 있는 Femur 프레임워크로 이 문제를 해결함.

 - Accelerate Distributed Joins with Predicate Transfer
   - Distributed Join, Predicate Transfer, Bloom Filter, Query Optimization, Cost-based Execution
   - 기존 predicate transfer는 단일 스레드에 국한되며 분산 환경에서 오히려 오버헤드만 발생하는 문제가 있음.
   - 분산 환경에 맞는 코스트 기반 실행과 불필요한 전송 제거를 통해 이 문제를 해결함.

 - Galley: Modern Query Optimization for Sparse Tensor Programs
   - Sparse Tensors, Declarative Programming, Cost-based Optimization, Tensor Compiler, Graph Algorithms
   - 기존 시스템은 희소 텐서 연산의 순서와 형식을 수동으로 지정해야 하며 최적화가 어렵고 번거로운 문제가 있음.
   - 선언형 희소 텐서 언어와 비용 기반 최적화를 통해 자동으로 실행 계획을 생성하고 성능 문제를 해결함.

 - Table Overlap Estimation through Graph Embeddings
   - Table Deduplication, Table Overlap, Graph Neural Network, Table Embedding, Similarity Estimation
   - 기존 중첩 추정 기법은 계산 시간이 과도하게 오래 걸려 확장성에 심각한 문제가 있음.
   - 그래프 임베딩 기반 GNN 모델을 통해 중첩 비율을 근사하여 빠르게 추정함으로써 이 문제를 해결함.

 - Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation
   - Retrieval-Augmented Generation (RAG), KV-Cache, Chunk Reuse, LLM Efficiency, Prompt Optimization
   - 기존 RAG 시스템은 동일한 텍스트 청크에 대해 매번 KV를 재계산하여 계산 낭비와 지연이 발생하는 문제가 있음.
   - KV 재사용 가능한 청크 캐시를 관리하고 출력 품질을 유지하며 효율적으로 재계산 및 캐시 관리를 수행하여 이 문제를 해결함.

 - MIRAGE-ANNS: Mixed Approach Graph-based Indexing for Approximate Nearest Neighbor Search
   - Approximate Nearest Neighbor Search (ANNS), Graph-based Indexing, HNSW, K-Graph, Incremental Insertion
   - 기존 그래프 인덱스는 빠른 구축 속도와 높은 검색 성능을 동시에 만족시키기 어렵고 점진적 삽입이 제한되는 문제가 있음.
   - 구축 속도와 검색 성능을 모두 만족하고 점진적 삽입도 지원하는 혼합형 MIRAGE-ANNS 방식으로 이 문제를 해결함.

 - A New Paradigm in Tuning Learned Indexes: A Reinforcement Learning-Enhanced Approach
   - Learned Index Structures (LIS), Reinforcement Learning, Auto-Tuning, Online Adaptation, Index Optimization
   - 기존 LIS는 수동 튜닝 부담이나 고정 설정의 한계로 인해 특정 워크로드에 최적화하기 어려운 문제가 있음.
   - 강화학습 기반 자동 튜닝 프레임워크 LITune을 통해 환경 변화에 따라 효율적으로 동적 조정함으로써 이 문제를 해결함.

 - Accelerating Graph Indexing for ANNS on Modern CPUs
   - Approximate Nearest Neighbor Search (ANNS), Graph Indexing, HNSW, SIMD, Vector Compression
   - 그래프 기반 ANNS는 검색 성능은 뛰어나지만 인덱싱 시간이 길어 대규모 데이터 처리에 어려움이 있는 문제가 있음.
   - SIMD 최적화 및 압축 코딩 전략 Flash를 적용하여 인덱스 구축 속도를 대폭 개선함으로써 이 문제를 해결함.

 - Automated Validating and Fixing of Text-to-SQL Translation with Execution Consistency
   - Text-to-SQL, Dataset Quality, Execution Consistency, SQL Equivalence, Error Detection
   - 기존 Text-to-SQL 데이터셋은 NL-SQL 매핑 오류가 많아 모델 정확도에 부정적인 영향을 주는 문제가 있음.
   - 실행 결과 기반 일관성 검사를 통해 오류를 검출하고 수정하는 SQLDriller를 통해 이 문제를 해결함.

 - CARINA: An Efficient CXL-Oriented Embedding Serving System for Recommendation Models
   - Embedding-based Recommendation, CXL, DRAM-CXL Hybrid Memory, Bandwidth-Aware Scheduling, NUMA Optimization
   - 기존 ERM 시스템은 CXL의 낮은 대역폭과 NUMA 구조로 인해 병목과 성능 저하 문제가 있음.
   - 핫 임베딩 분산 배치 및 대역폭 인지형 태스크 스케줄링을 통해 이 문제를 해결함.

 - Data Enhancing for Machine Learning (Data Enhancement for Binary Classification of Relational Data)
   - Adversarial Robustness, Data Poisoning, Feature Debugging, Data Augmentation, Binary Classification
   - 기존 학습 데이터는 독성 특징이나 보이지 않는 공격에 취약해 분류기의 견고성에 문제가 있음.
   - 데이터 정제 및 적대적 예제 추가를 통해 정확도 유지하면서도 견고성과 처리 속도를 함께 향상시킴.
   
 - PLM4NDV: Minimizing Data Access for Number of Distinct Values Estimation with Pre-trained Language Models
   - NDV Estimation, Pre-trained Language Models (PLM), Schema Semantics, Data Access Reduction, Learned Estimation
   - 기존 NDV 추정은 전체 컬럼 접근이나 많은 샘플이 필요해 데이터 접근 비용이 큰 문제가 있음.
   - 스키마 의미를 PLM으로 추출해 데이터 접근 없이도 NDV를 정확히 예측함으로써 이 문제를 해결함.

 - GPU-Accelerated Graph Cleaning with a Single Machine (Rule-Based Graph Cleaning with GPUs on a Single Machine)
   - Graph Cleaning, Rule-based Processing, GPU Acceleration, CPU–GPU Synergy, Single-Machine Optimization
   - 기존 단일 머신 시스템은 계산량과 I/O 부담이 큰 그래프 정제를 감당하지 못하는 문제가 있음.
   - CPU–GPU 파이프라인, 메모리 최적화, 다중 병렬 모델을 결합한 MiniClean 시스템으로 이 문제를 해결함.

 - Rethinking The Compaction Policies in LSM-trees
   - LSM-tree, Compaction Policy, Write/Read Amplification, Dynamic Programming, Query Throughput
   - 기존 compaction 정책은 WA와 RA 사이의 단순 트레이드오프에 집중하여 자원 활용 최적화가 부족한 문제가 있음.
   - 질의 처리량 극대화를 위한 동적 계획 기반의 EcoTune 알고리즘으로 compaction을 자원 투자 관점에서 재설계함.

 - PQCache: Product Quantization-based KVCache for Long Context LLM Inference
   - KVCache Compression, Product Quantization (PQ), LLM Inference, Long Context, Self-Attention Optimization
   - 기존 KVCache 최적화 기법은 모델 품질 저하나 지연 증가로 인해 추론 효율에 한계가 있는 문제가 있음.
   - PQ 기반 압축과 근사 검색으로 품질을 유지하면서 지연을 줄이는 PQCache 방식으로 이 문제를 해결함

 - PrivPetal: Relational Data Synthesis via Permutation Relations
   - Differential Privacy, Synthetic Relational Data, Permutation Relation, Foreign Key, Markov Random Field
   - 기존 방식은 관계형 구조를 표현하기 위해 많은 프라이버시 예산을 소모하는 EM 기반 합성이 문제가 있음.
   - 순열 관계와 마르코프 랜덤 필드를 활용해 고차원 관계를 우회적으로 생성함으로써 이 문제를 해결함.

 - A Structured Study of Multivariate Time-Series Distance Measures
   - Multivariate Time Series, Distance Measures, Normalization, Sliding vs Elastic, Statistical Analysis
   - 기존 연구는 lock-step과 elastic에 편중되고 정규화와 통계 분석 측면에서도 한계가 있는 문제가 있음.
   - 다양한 계열 거리 측정과 정규화를 체계적으로 비교·분석하여 설계와 선택에 대한 가이드를 제시함.

 - OpenSearch-SQL: Enhancing Text-to-SQL with Dynamic Few-shot and Consistency Alignment
   - Text-to-SQL, Consistency Alignment, SQL-Like Language, Query-CoT-SQL, Large Language Model
   - 기존 LLM 기반 방식은 지시 실패나 환각 등으로 인해 Text-to-SQL 성능에 한계가 있는 문제가 있음.
   - 일관성 정렬, 중간 언어, 동적 few-shot 전략을 통합한 모듈형 프레임워크로 이 문제를 해결함

 - Maximus: A Modular Accelerated Query Engine for Data Analytics on Heterogeneous Systems
   - Heterogeneous Hardware, Modular Query Engine, GPU Acceleration, Operator-Level Integration, Data Pipeline Optimization
   - 기존 데이터 처리 시스템은 이기종 하드웨어와 조립형 소프트웨어 구조에 적응하기 어려운 문제가 있음.
   - CPU-GPU 병렬 실행, 오퍼레이터 통합, 파이프라인 최적화를 갖춘 모듈형 Maximus 엔진으로 이 문제를 해결함.

 - Approximate DBSCAN under Differential Privacy
   - Differential Privacy, DBSCAN, Span-based Clustering, Histogram Construction, Privacy-Preserving Clustering
   - 기존 방식은 클러스터 레이블을 직접 출력하므로 프라이버시 하에서 유용성이 거의 없는 문제가 있음.
   - 스팬 정보를 기반으로 한 새로운 정의와 선형 시간 알고리즘으로 이 문제를 해결함.

 - Auto-Test: Learning Semantic-Domain Constraints for Unsupervised Error Detection in Tables
   - Data Cleaning, Error Detection, Table Constraints, Constraint Learning, Semantic Domain
   - 기존 정제 기법은 테이블마다 전문가가 제약 조건을 수작업으로 지정해야 하는 문제가 있음.
   - 자동 학습된 의미 기반 제약 조건을 활용해 오류를 탐지하고 전문가 방식과 병행 사용 가능하게 하여 이 문제를 해결함.
   
 - Approximating Opaque Top-k Queries
   - Top-k Query, Opaque UDF, Bandit Algorithm, Hierarchical Index, Approximate Query Processing
   - 불투명한 scoring 함수로 인해 정확한 top-k 질의 평가가 비용이 많이 들고 인덱싱이 불가능한 문제가 있음.
   - 계층 인덱스와 fat-tail을 공략하는 밴딧 알고리즘으로 근사 top-k 성능을 빠르게 달성하여 이 문제를 해결함.

 - Nested Parquet Is Flat, Why Not Use It? How To Scan Nested Data With On-the-Fly Key Generation and Joins.
   - Parquet, Nested Data, Columnar Format, On-the-Fly Join, Query Acceleration
   - 중첩 데이터를 지원하지 않거나 성능이 낮은 기존 쿼리 엔진의 한계로 실질적 사용이 어려운 문제가 있음.
   - 실시간 키 생성 및 조인을 통해 Parquet에서 직접 중첩 구조를 재구성하는 방식으로 이 문제를 해결함.

 - Incremental Rule Discovery in Response to Parameter Updates
   - Rule Mining, Support & Confidence Thresholds, Incremental Algorithms, Entity Enhancing Rules, Parameter Tuning
   - 임계값 변경 시 매번 규칙을 처음부터 재탐색해야 하므로 계산 비용이 큰 문제가 있음.
   - 변경된 부분만 계산하는 최적 증분 알고리즘을 통해 이 문제를 해결함.
   - **흥미로운 or 신기한**

 - Clementi: Efficient Load Balancing and Communication Overlap for Multi-FPGA Graph Processing
   - Graph Processing, Multi-FPGA, Load Balancing, Communication Overlap, Fine-Grained Pipeline
   - 기존 다중 FPGA 시스템은 통신 오버헤드와 작업 불균형으로 확장성과 성능에 한계가 있는 문제가 있음.
   - 통신-계산 중첩과 부하 균형 기반의 파이프라인 구조로 이 문제를 해결함.

 - T3: Accurate and Fast Performance Prediction for Relational Database Systems With Compiled Decision Trees
   - Query Performance Prediction, Decision Tree, Pipeline Modeling, Tuple-Centric Estimation, Low Latency
   - 기존 예측기는 정확하지만 추론이 느려 실시간 적용이 어려운 문제가 있음.
   - 파이프라인 단위 예측과 튜플 단위 추정을 적용한 빠르고 정확한 트리 기반 모델 T3로 이 문제를 해결함.

 - Dupin: A Parallel Framework for Densest Subgraph Discovery in Fraud Detection on Massive Graphs
   - Densest Subgraph Discovery (DSD), Fraud Detection, Graph Processing, Parallel Framework, Billion-Scale Graphs
   - 기존 DSD 방식은 순차 처리로 인해 대규모 트랜잭션 그래프에서 성능과 탐지 지연 문제가 있음.
   - 병렬화된 DSD 처리와 사용자 맞춤 목표 지원을 통해 빠르고 정확한 탐지를 가능하게 함.

 - Divide-and-Conquer: Scalable Shortest Path Counting on Large Road Networks
   - Shortest Path Counting, Road Networks, Divide-and-Conquer, 2-hop Labeling, Graph Partitioning
   - 기존 방법은 모든 최단 경로를 세는 데 확장성과 저장 공간 문제로 어려움이 있는 문제가 있음.
   - 정점 이분 기반 분할 정복과 재구성 정리를 통해 계산 효율성과 확장성을 크게 높여 이 문제를 해결함.

 - Zombie Hashing: Reanimating Tombstones in Graveyard
   - Linear Probing, Hash Table, Primary Clustering, Deamortization, Tombstone Reuse
   - 기존 선형 탐사 해시 테이블은 부하율이 높아지면 클러스터링 현상으로 인해 성능이 급격히 저하되는 문제가 있음.
   - Tombstone을 소규모 윈도우 내에서 점진적으로 재배치하여, 성능 저하 없이 고부하에서도 일관된 성능을 유지함.

 - LpBound: Pessimistic Cardinality Estimation Using Lp-Norms of Degree Sequences
   - Cardinality Estimation, Query Optimizer, Join Queries, ℓₚ-Norms, Linear Programming
   - 기존 쿼리 옵티마이저는 부정확한 카디널리티 추정으로 인해 잘못된 실행 계획을 선택하는 문제가 있음.
   - ℓₚ-노름 기반 통계를 활용해 선형계획법으로 상한을 계산하는 LpBound 추정기로 이 문제를 해결함.

 - Logical and Physical Optimizations for SQL Query Execution over Large Language Models
   - Query Optimization, Large Language Models (LLMs), SQL over LLMs, GALOIS System, Result Quality
   - 기존 SQL 최적화는 LLM의 문맥 이해 부족으로 인해 정확도가 낮아지는 문제가 있음.
   - LLM 특화 연산자 및 메타데이터 수집으로 품질과 비용을 균형 있게 개선하는 시스템(GALOIS)으로 해결함.

 - Privacy and Accuracy-Aware AI/ML Model Deduplication
   - Differential Privacy, Model Deduplication, DP-SGD, Compression, Inference Efficiency
   - 기존 모델 중복 제거는 DP 기반 모델에 적용 시 프라이버시 비용 누적과 효율성 저하 문제가 있음.
   - 프라이버시 및 정확도 인식 전략과 Sparse Vector 기법으로 효율적 중복 제거를 달성함.

 - Graph Edit Distance Estimation: A New Heuristic and A Holistic Evaluation of Learning-based Methods
   - Graph Edit Distance (GED), Heuristic Algorithm, Graph Neural Networks, Interpretable Models
   - 기존 GED 예측 모델은 분야 간 평가가 부족하고 단일 휴리스틱만을 기준으로 비교하는 문제가 있음.
   - 간단하고 해석 가능한 휴리스틱(App-BMao)을 제안하여 기존 학습 기반 모델보다 더 정확하게 GED를 추정함.

 - HotStuff-1: Linear Consensus with One-Phase Speculation
   - Byzantine Fault Tolerance (BFT), Consensus Protocol, Leader Rotation, Speculation
   - 기존 BFT 합의 프로토콜은 네트워크 지연과 리더 지연 문제로 인한 성능 저하 문제가 있음.
   - 조기 투기 확정 및 슬롯 기반 리더 제어로 네트워크 지연과 리더 비협조 문제를 해결함.
   
 - Accelerating Skyline Path Enumeration with a Core Attribute Index on Multi-attribute Graphs
   - Skyline Query, Multi-Attribute Graphs, s-t Path Enumeration, Indexing
   - 기존 경로 나열 기법은 다중 속성 그래프에서 skyline 조건을 효율적으로 처리하지 못하는 문제가 있음.
   - 핵심 속성 인덱스(CAI)와 병렬 레이블 전파를 통해 skyline 경로 탐색을 가속화하여 이 문제를 해결함.

 - Efficient Dynamic Indexing for Range Filtered Approximate Nearest Neighbor Search
   - Approximate Nearest Neighbor (ANN), Range Filter, Product Quantization (PQ), Indexing, High-dimensional Search
   - 기존 ANN 검색은 범위 조건을 함께 처리할 때 저장 공간과 성능 간의 트레이드오프 문제가 있음.
   - 범위 필터와 갱신을 효율적으로 지원하는 새로운 색인 구조 RangePQ로 이 문제를 해결함.

 - PrivRM: A Framework for Range Mean Estimation under Local Differential Privacy
   - Local Differential Privacy (LDP), Range Query, Mean Estimation, Numerical Perturbation
   - 기존 LDP 방식은 전체 도메인 평균에 초점을 두어 범위 평균 추정 시 과도한 노이즈 문제가 있음.
   - 범위 평균 추정을 위해 노이즈를 동적으로 조정하는 프레임워크 PrivRM으로 이 문제를 해결함.

 - SuSe: Summary Selection for Regular Expression Subsequence Aggregation over Streams
   - Stream Processing, Regular Expression, Complex Event Processing (CEP), Query Optimization, Data Summarization
   - 스트림 데이터에서 정규 표현식으로 정의된 하위 시퀀스에 대한 집계 쿼리 처리 시, 단일 요약(summary) 구조로는 다양한 형태의 정규 표현식에 대해 최적의 성능을 내기 어려움.
   - 여러 종류의 요약 구조를 미리 유지하고, 쿼리가 주어지면 해당 정규 표현식의 특성을 분석하여 가장 효율적으로 처리할 수 있는 최적의 요약 구조를 동적으로 선택함.

 - Serf: Streaming Error-Bounded Floating-Point Compression
   - Regular Expressions, Subsequence Matching, Stream Processing, Aggregation, Pattern Matching
   - 기존 정규표현식 엔진은 Kleene closure를 포함한 부분수열 패턴에서 매칭 수가 많아 처리가 어려운 문제가 있음.
   - 전체 매칭 대신 요약된 상태 정보를 기반으로 집계를 수행하는 SuSe 아키텍처로 이 문제를 해결함.

 - Interactive Graph Search Made Simple
   - Graph Search, User Interaction, Information Retrieval, Algorithm Design
   - 기존 IGS 알고리즘은 상호작용 최적화가 어렵고 사용자 입력 요구가 많아 사용이 어려운 문제가 있음.
   - 블랙박스 연산 기반의 프레임워크와 CPU 시간을 줄이는 알고리즘으로 이 문제를 해결함.

 - SHIELD: Encrypting Persistent Data of LSM-KVS from Monolithic to Disaggregated Datacenters
   - LSM-KVS, Persistent Encryption, Disaggregated Storage, WAL, RocksDB
   - 기존 LSM-KVS는 저장 데이터에 대한 암호화 성능이 낮고 DS 환경에 적합하지 않은 문제가 있음.
   - LSM 구조에 통합된 SHIELD 설계로 성능 손실 없이 암호화와 확장성을 동시에 해결함.

 - How to Grow an LSM-tree: Towards Bridging The Gap Between Theory and Practice
   - LSM-tree, Vertical Growth, Horizontal Growth, Write Optimization, RocksDB
   - 기존 LSM 확장 방식은 읽기-쓰기 성능이나 공간 비용 측면에서 균형을 이루기 어려운 문제가 있음.
   - 수직과 수평 확장의 장점을 결합한 Vertiorizon 기법으로 성능과 공간 비용 문제를 동시에 해결함.
   
 - Fast Approximate Similarity Join in Vector Databases
   - Vector Databases, Similarity Join, Embedding, k-Similarity, Proximity Graph
   - 기존 유사 조인 방식은 질의 기반으로 설계되어 조인 연산 고유 특성을 활용하지 못하는 문제가 있음.
   - 조인 윈도우 간 관계를 활용한 SimJoin 알고리즘으로 조인 속도와 효율성을 향상시킴.

 - Extending SQL to Return a Subdatabase
   - SQL, Subdatabase, Normal Forms, ResultDB, Query Semantics
   - SQL은 단일 테이블만 결과로 반환하여 중복과 비정규화 문제를 유발하는 문제가 있음.
   - 여러 테이블을 관련 튜플만 포함된 서브데이터베이스로 반환하도록 RESULTDB 확장으로 해결함.

 - RLOMM: An Efficient and Robust Online Map Matching Framework with Reinforcement Learning
   - Online Map Matching, Reinforcement Learning, OMDP, Contrastive Learning, Real-time Trajectory
   - 기존 지도 매칭 기법은 대규모 환경에서 정확도와 효율, 강건성을 동시에 만족하지 못하는 문제가 있음.
   - OMDP 기반 모델링과 강화학습, 대조학습을 활용한 표현 정렬로 정확도와 효율, 강건성을 동시에 해결함.

 - SPACE: Cardinality Estimation for Path Queries Using Cardinality-Aware Sequence-based Learning
   - Cardinality Estimation, Path Pattern, Graph Database, Sequence Encoding, Machine Learning
   - 그래프 경로 질의의 카디널리티 추정에 대한 연구가 부족하고 정확도와 일반화에 한계가 있는 문제가 있음.
   - 시퀀스 인코딩과 카디널리티 인식 이중 인코딩을 통해 그래프 경로의 구조와 분포를 효과적으로 학습하여 정확도를 해결함.

 - Dangers of List Processing in Querying Property Graphs
   - Property Graphs, Pattern Matching, List Processing, Cypher, Query Optimization
   - 리스트 기반 후처리가 높은 표현력을 유도하여 작은 그래프에서도 질의 시간 초과 문제가 있음.
   - 리스트 처리의 유용성은 유지하면서도 비현실적인 질의를 방지하는 제한적 사용 가이드를 제안해 해결함.

 - Athena: An Effective Learning-based Framework for Query Optimizer Performance Improvement
   - Query Optimization, Learned Models, Execution Plan Exploration, PostgreSQL, Cost Models
   - 기존 학습 기반 최적화기는 계획 탐색 공간이 제한되고 실행 계획으로부터 비효율적으로 학습되는 문제가 있음.
   - 순서 중심 탐색기와 Tree-Mamba 비교기 및 시간 가중 손실 함수를 통해 학습 효율을 개선하여 이 문제를 해결함.
   - **흥미로운 or 신기한**

 - Efficient Indexing for Flexible Label-Constrained Shortest Path Queries in Road Networks
   - Label-Constrained Shortest Path, Regular Language, Road Network, Indexing, Query Optimization
   - 기존 라벨 제약 최단 경로 질의는 언어 제약이 비유연하고 질의 처리 속도가 느린 문제가 있음.
   - 경계 정점을 활용한 인덱스를 통해 탐색을 건너뛰고 언어 제약을 유연하게 처리함으로써 이 문제를 해결함.

 - MatCo: Computing Match Cover of Subgraph Query over Graph Data
   - Subgraph Query, Match Cover, Graph Database, Search Space Reduction, Pattern Matching
   - 기존 서브그래프 질의는 중복된 매치가 많아 탐색 공간이 크고 비효율적인 문제가 있음.
   - 정점 커버만 만족하는 부분 결과 집합을 선별하고 불필요한 탐색을 줄이는 프레임워크(MatCo)로 이 문제를 해결함.

 - GPH: An Efficient and Effective Perfect Hashing Scheme for GPU Architectures
   - GPU Hash Table, Perfect Hashing, Lookup Performance, Micro-Benchmark, Parallelism
   - 기존 GPU 해시 테이블은 조회 성능에 대한 체계적인 분석이 부족하고 최적화가 어려운 문제가 있음.
   - 완전 해싱 기법과 병렬 최적화를 적용한 GPH로 GPU 해시 테이블의 조회 성능을 극대화함.

 - Fast and Scalable Data Transfer across Data Systems
   - Data Transfer, Heterogeneous Systems, Middleware, Scalability, XDBC Framework
   - 기존 데이터 전송 방식은 이기종 시스템과 환경에서 속도와 확장성을 동시에 만족시키기 어려운 문제가 있음.
   - 논리적 구성요소와 다양한 구현을 분리한 XDBC 프레임워크로 환경에 맞는 최적 구성을 통해 전송 문제를 해결함.

 - Credible Intervals for Knowledge Graph Accuracy Estimation
   - Knowledge Graphs, Accuracy Estimation, Bayesian Statistics, Credible Intervals, aHPD Algorithm
   - 기존 KG 정확도 추정 방식은 빈도주의 기반 신뢰구간을 사용하여 해석 오류 가능성이 있는 문제가 있음.
   - 베이지안 기반 신뢰구간(CrI)과 적응형 알고리즘(aHPD)을 활용하여 정확도 추정의 신뢰성과 해석 가능성을 해결함.

 - DFlush: DPU-Offloaded Flush for Disaggregated LSM-based Key-Value Stores
   - LSM-KVS, Flush Optimization, DPU Offloading, CPU Reduction, Tail Latency
   - 기존 LSM-KVS는 플러시 작업이 CPU 자원을 과도하게 소모하여 성능 저하 문제가 있음.
   - 플러시 작업을 DPU에 오프로딩하고 우선순위 기반 스케줄링으로 CPU 사용과 tail latency 문제를 해결함.

 - Relevance queries for interval data
   - Interval Data, Relevance Score, Overlap Query, Interval Index, Temporal Databases
   - 단순 겹침 기반 질의는 결과가 과도하게 많아 연관도 기반 필터링이 어려운 문제가 있음.
   - 상대적 겹침 정도를 이용한 연관도 질의 방식과 인덱스 기반 평가 프레임워크로 이 문제를 해결함.

 - Computing Inconsistency Measures Under Differential Privacy
   - Data Quality, Inconsistency Measure, Conflict Graph, Differential Privacy, Private Graph Statistics
   - 기존 불일치 측정값들은 민감도 문제로 인해 차등 개인정보 보호 환경에서 계산이 어려운 문제가 있음.
   - conflict graph 기반 추정 기법과 DP 그래프 통계로 이 문제를 해결함.

 - NEXT: A New Secondary Index Framework for LSM-based Data Storage
   - LSM-tree, Secondary Index, Non-key Attributes, Query Performance, RocksDB
   - 비키 속성 질의를 위한 기존 보조 인덱스 구조는 성능과 일관성 면에서 한계가 있음.
   - NEXT는 인덱스 블록과 글로벌 인덱스를 활용한 2단계 구조로 성능과 일관성을 개선함.

 - Apt-Serve: Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving
   - LLM Inference, Serving System, KV Cache, Request Scheduling, Hybrid Cache
   - 기존 LLM 서빙 시스템은 정적인 캐시 관리와 휴리스틱 기반 스케줄링으로 인해, 동적인 요청 환경에서 GPU 메모리 병목 현상 및 성능 저하를 겪고 있음.
   - CPU-GPU 하이브리드 캐시를 도입하고, 실시간 캐시 적중률과 GPU 메모리 상태를 반영하여 동적으로 요청 배치와 우선순위를 정하는 적응형 스케줄링 방식을 제안함.

 - Adda: Towards Efficient in-Database Feature Generation via LLM-based Agents
   - Feature Engineering, In-Database Analytics, LLM, Agents, Code Generation, UDF
   - 머신러닝의 피처 엔지니어링은 전문가의 지식과 많은 시간을 요구하는 시행착오 과정이며, 데이터를 데이터베이스 외부로 이동시켜 처리하는 비효율성을 가지고 있음.
   - 자연어 설명을 LLM 기반 에이전트가 해석하여 피처 생성 코드를 만들고, 이를 SQL로 변환 후 UDF로 컴파일하여 데이터베이스 내에서 직접 피처를 생성하는 방식을 제안함.

 - RWalks: Random Walks as Attribute Diffusers for Filtered Vector Search
   - Filtered Vector Search, Attribute Filtering, Graph-based Index, Random Walks, HNSW
   - 필터가 포함된 벡터 검색 시, 결과를 모두 찾은 후 필터링하는 방식은 필터의 선택도가 높을 때 매우 비효율적인 문제를 가지고 있음.
   - 랜덤 워크를 통해 벡터 유사도 그래프에 속성 정보를 미리 '확산'시켜, 검색 시 필터를 만족할 만한 후보로 탐색 경로를 유도하는 방식을 제안함.
   - **흥미로운 or 신기한**

 - SpareLLM: Automatically Selecting Task-Specific Minimum-Cost Large Language Models under Equivalence Constraint
   - Large Language Models (LLM), Model Selection, Cost Optimization, Equivalence Testing, MLOps
   - 특정 작업에 요구되는 품질을 충족하는 가장 저렴한 LLM을 찾는 과정이 수동적이고, 시간과 비용이 많이 소요되는 어려움을 가지고 있음.
   - 참조 모델과 결과가 통계적으로 동일한 모델 중 가장 저렴한 모델을 자동으로 찾는 2단계 통계 기반 최적화 프레임워크를 제안함.

 - Fair and Actionable Causal Prescription Ruleset
   - Causal Inference, Algorithmic Fairness, Prescriptive Analytics, Actionability, Rule-based Models
   - 기존의 처방 모델은 해석이 어렵고, 특정 인구 그룹에 불공정하며, 비현실적인 행동을 추천하는 경향이 있음.
   - 규칙의 인과적 효과를 극대화하면서 공정성과 실행 가능성 제약을 직접 최적화 문제에 포함시켜, 해석 가능하고 공정하며 실행 가능한 'IF-THEN' 규칙 집합을 생성하는 프레임워크를 제안함.

 - Community Detection in Heterogeneous Information Networks Without Materialization
   - Community Detection, Heterogeneous Information Networks (HIN), Meta-path, Materialization, In-Database Analytics, Query-driven
   - HIN에서 커뮤니티를 찾기 위해 특정 관계(메타-경로)를 기반으로 새로운 그래프를 생성하는 과정은 막대한 저장 공간과 계산 비용을 필요로 함.
   - 그래프를 미리 생성하지 않고, 데이터베이스 상에서 직접 SQL 쿼리를 통해 커뮤니티 탐색을 수행하여 구체화(materialization)에 따르는 비용을 제거하는 방식을 제안함.

 - Mnemosyne: Dynamic Workload-Aware BF Tuning via Accurate Statistics in LSM trees
   - LSM-tree, Bloom Filter, Dynamic Tuning, Workload-Aware, RocksDB, Performance Optimization
   - LSM-tree에서 사용하는 블룸 필터의 메모리를 정적으로 할당하여, 워크로드의 특성을 반영하지 못하고 메모리 낭비나 I/O 비효율을 겪고 있음.
   - 런타임의 쿼리 패턴 통계를 수집하고, 이를 기반으로 자주 접근하는 데이터의 블룸 필터에 더 많은 메모리를 동적으로 할당하는 비용 기반 최적화 방식을 제안함

 - Are database system researchers making correct assumptions about transaction workloads?
   - Transaction Workloads, OLTP, Database Benchmarks, Concurrency Control, Contention, Skew Analysis
   - 데이터베이스 연구에서 사용하는 TPC-C, YCSB 같은 표준 벤치마크들이 실제 운영 환경의 트랜잭션 특성(특히 데이터 접근 편향)을 제대로 반영하지 못하고 있음.
   - 실제 대규모 서비스의 트랜잭션 데이터를 분석하여, 실제 워크로드는 벤치마크보다 훨씬 높은 데이터 접근 편향(skew)을 가짐을 밝혀내고, 연구 커뮤니티에 보다 현실적인 워크로드 모델의 필요성을 제시함.

 - Learned Offline Query Planning via Bayesian Optimization
   - Query Optimization, Learned Query Planning, Bayesian Optimization, Offline Learning, Black-box Optimization
   - 전통적인 쿼리 옵티마이저는 부정확한 비용 모델로 인해 최적의 실행 계획을 찾지 못하는 경우가 많고, 학습 기반 옵티마이저는 온라인 학습으로 인한 부담이 있음.
   - 실제 쿼리 실행 시간을 블랙박스 함수로 보고, 베이지안 최적화를 사용해 오프라인 환경에서 직접 쿼리를 실행하며 최적의 실행 계획을 학습하는 방식을 제안함.
   - **흥미로운 or 신기한**

 - SPARTAN: Data-Adaptive Symbolic Time-Series Approximation
   - Time-Series, Symbolic Approximation, SAX, Data-Adaptive, Breakpoint Selection
   - 대표적인 시계열 기호화 방법인 SAX는 데이터가 정규분포를 따른다고 가정하고 기호를 정의하여, 실제 데이터의 분포 특성을 제대로 반영하지 못하는 문제가 있음.
   - 데이터의 실제 분포를 기반으로 기호화의 기준이 되는 구분점(breakpoint)을 학습하여, 원본 시계열의 정보를 더 잘 보존하는 데이터 적응적인 기호화 방식을 제안함.

 - Understanding the Black Box: A Deep Empirical Dive into Shapley Value Approximations for Feature Explanations
   - Explainable AI (XAI), Shapley Value, Feature Importance, Approximation Algorithms, Model Interpretability, Empirical Study
   - 섀플리 값 근사 알고리즘들은 널리 쓰이지만, 각 방법의 정확도, 비용, 안정성에 대한 체계적인 비교나 선택 기준이 부족함.
   - 다양한 모델과 데이터에 대해 여러 섀플리 값 근사 알고리즘의 성능을 실증적으로 비교 분석하여, 특정 상황에 어떤 알고리즘이 적합한지에 대한 실용적인 가이드라인을 제시함.

 - Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System
   - Tabular Data Retrieval, Semantic Search, Large Language Models (LLM), Data Discovery, Vector Embedding
   - 기존의 테이블 데이터 검색은 키워드 매칭에 의존하여, 데이터의 의미를 이해하지 못하고 검색 정확도가 떨어지는 한계를 가지고 있음.
   - LLM을 이용해 각 테이블의 종합적인 설명문을 만들고 이를 벡터로 변환하여, 자연어 질의와의 의미적 유사도를 기반으로 테이블을 검색하는 방식을 제안함.

 - Low Rank Learning for Offline Query Optimization
   - Query Optimization, Cardinality Estimation, Learned Optimizer, Low-Rank Learning, Matrix Completion, Offline Learning
   - 학습 기반 쿼리 옵티마이저는 모델이 복잡하고 파라미터가 많아, 학습에 많은 데이터와 시간이 필요하며 과적합되기 쉬운 문제를 가지고 있음.
   - 쿼리들의 카디널리티 관계를 '저차원 행렬'로 간주하고, 행렬 완성(matrix completion) 기법을 통해 적은 데이터로도 효율적으로 학습하는 방식을 제안함.
   - **흥미로운 or 신기한**

 - Fast Hypertree Decompositions via Linear Programming: Fractional and Generalized
   - Hypertree Decomposition, Conjunctive Queries, Query Optimization, Linear Programming, Fractional Width
   - 쿼리 구조의 복잡도를 나타내는 최적의 하이퍼트리 분해를 찾는 것은 계산 비용이 매우 높은 NP-hard 문제를 가지고 있음.
   - 하이퍼트리 분해 문제를 선형 계획법(LP)으로 모델링하고, 이를 풀어 얻은 소수점 해(fractional solution)를 정수 해로 변환하여 빠르고 효율적으로 분해를 찾는 방식을 제안함.

 - Mitigating the Impedance Mismatch between Prediction Query Execution and Database Engine
   - In-Database ML, Prediction Query, Impedance Mismatch, UDF, Query Optimization, Relational Operators
   - 데이터베이스의 집합 단위 처리 방식과 ML 예측의 행 단위 처리 방식 간의 '임피던스 불일치'로 인해 UDF를 통한 예측 쿼리 실행이 비효율적인 문제를 가지고 있음.
   - ML 모델의 예측 과정을 데이터베이스 친화적인 관계형 연산자들의 조합으로 변환하여, 쿼리 옵티마이저가 데이터 처리와 예측 과정을 함께 최적화할 수 있도록 하는 방식을 제안함.

 - DIGRA: A Dynamic Graph Indexing for Approximate Nearest Neighbor Search with Range Filter
   - Approximate Nearest Neighbor Search (ANNS), Filtered Search, Dynamic Indexing, Graph-based Index, Range Filter
   - 기존 그래프 기반 ANNS 인덱스는 데이터가 계속 변하는 동적 환경에서 속성 필터를 효율적으로 처리하지 못하는 문제를 가지고 있음.
   - 속성 값의 범위를 기준으로 그래프를 여러 서브그래프로 분할하고, 데이터의 삽입/삭제를 효율적으로 처리하며 필터 조건에 맞는 서브그래프만 탐색하는 동적 인덱스 구조를 제안함.

 - Cracking SQL Barriers: An LLM-based Dialect Translation System
   - SQL Dialect Translation, Code Generation, Large Language Models (LLM), Database Interoperability, Knowledge-Augmented Generation
   - 데이터베이스마다 다른 SQL 방언(dialect)으로 인해 쿼리를 수동으로 번역해야 하며, 기존 번역기는 복잡하고 미묘한 차이를 처리하지 못하는 한계를 가지고 있음.
   - LLM을 사용하되, 번역 전 쿼리를 분석하고 방언 차이에 대한 지식 베이스의 정보를 결합한 프롬프트를 생성하여 정확도를 높이는 지식 증강 번역 시스템을 제안함.

 - A Cost-Effective LLM-based Approach to Identify Wildlife Trafficking in Online Marketplaces (Data-Driven Applications)
   - Wildlife Trafficking, Online Marketplaces, LLM, Cost-Effectiveness, Classification Cascade, Text Mining
   - 온라인상의 야생동물 불법 거래는 은어를 사용해 탐지가 어렵고, 이를 해결할 수 있는 고성능 LLM은 모니터링에 사용하기엔 비용이 너무 비싼 문제를 가지고 있음.
   - 저비용 규칙 기반 필터와 소형 LLM을 통해 의심스러운 게시물을 단계적으로 거른 뒤, 최종 후보군만 고성능 LLM으로 판별하는 다단계 필터링 방식을 제안함.

 - PDX: A Data Layout for Vector Similarity Search
   - Vector Similarity Search, Data Layout, Approximate Nearest Neighbor (ANN) Search, Quantization, HNSW
   - 기존 벡터 검색은 데이터 압축 시 정확도가 떨어지고, 정확한 거리 계산을 위해 압축을 해제하는 과정에서 비효율이 발생하는 문제를 가지고 있음.
   - 벡터 데이터 자체를 점진적으로 계산 가능하도록 구성하고, 검색 시 필요한 만큼만 사용하여 전체 압축 해제 없이도 효율적인 탐색이 가능한 데이터 레이아웃을 제안함.

 - Two Birds with One Stone: Efficient Deep Learning over Mislabeled Data through Subset Selection
   - Noisy Labels, Mislabeled Data, Data Subset Selection, Coreset Selection, Efficient Training, Robust Learning
   - 딥러닝 학습 시, 데이터의 노이즈(오류)는 모델 성능을 저하시키고, 데이터의 크기는 학습 속도를 저하시키는 두 가지 문제를 동시에 가지고 있음.
   - 학습 과정에서 데이터의 특성을 분석하여, 노이즈가 없고 대표성을 가지는 데이터의 부분집합을 선택하여 학습함으로써 성능과 효율을 동시에 높이는 방식을 제안함.

 - cuMatch: A GPU-based memory-efficient worst-case optimal join processing method for subgraph queries with complex patterns
   - Subgraph Matching, Worst-Case Optimal Join, GPU, Parallel Computing, Graph Analytics
   - 최적의 조인 알고리즘(WCO)은 불규칙한 메모리 접근 패턴으로 인해 GPU의 병렬 처리 능력을 제대로 활용하기 어려운 문제를 가지고 있음.
   - GPU에 친화적인 압축 데이터 구조(CAM)를 통해 조인을 수행하고 동적 워크로드 밸런싱을 적용하여, GPU의 병렬성을 극대화하는 방식을 제안함.

 - The Best of Both Worlds: On Repairing Timestamps and Attribute Values for Multivariate Time Series
   - Time Series, Data Cleaning, Data Repair, Imputation, Timestamp Correction, Multivariate Time Series
   - 시계열 데이터에서 발생하는 타임스탬프 오류와 값 오류를 개별적으로 처리하여, 두 오류 유형 간의 상호작용을 고려하지 못하고 복구 정확도가 떨어지는 문제가 있음.
   - 타임스탬프와 값의 결합 확률 분포를 모델링하여, 두 종류의 오류를 분리하지 않고 하나의 최적화 문제로 동시에 복구하는 통합된 프레임워크를 제안함.

 - TableDC: Deep Clustering for Tabular Data
   - Deep Clustering, Tabular Data, Representation Learning, Heterogeneous Data, Unsupervised Learning
   - 기존 딥 클러스터링 방법은 이미지 같은 동종 데이터에 맞춰져 있어, 수치형과 범주형이 섞인 테이블 데이터의 복잡한 특성을 제대로 학습하지 못하는 문제가 있음.
   - 테이블 데이터의 이기종 특징을 효과적으로 다루는 딥러닝 구조와, 데이터 정보 보존 및 군집 분리를 동시에 최적화하는 손실 함수를 결합한 프레임워크를 제안함.

 - BPF-DB: A Kernel-Embedded Transactional Database Management System For eBPF Applications
   - eBPF, In-Kernel Database, Transactional System, ACID, Concurrency Control, Operating Systems
   - 커널에서 실행되는 eBPF 프로그램은 여러 데이터에 대한 원자적(atomic) 연산, 즉 트랜잭션을 지원하는 효율적인 상태 관리 수단이 부재한 문제를 가지고 있음
   - eBPF 애플리케이션을 위해 커널 내부에 직접 내장되어 ACID 트랜잭션을 보장하는 데이터베이스 시스템을 구축하여, 커널-유저스페이스 전환 없이 효율적인 상태 관리가 가능한 방식을 제안함.

 - Agree to Disagree: Robust Anomaly Detection with Noisy Labels
   - Anomaly Detection, Noisy Labels, Semi-supervised Learning, Robustness, Co-training, Disagreement
   - 준지도 이상 탐지는 일부 주어진 레이블에 크게 의존하는데, 이 레이블에 노이즈가 섞여 있으면 모델이 잘못된 패턴을 학습하여 성능이 저하되는 문제가 있음.
   - 두 개 이상의 모델을 동시에 학습시키되, 노이즈로 의심되는 데이터에 대해서는 모델들이 서로 '불일치'하도록 유도하여 노이즈에 과적합되는 것을 방지하는 방식을 제안함.

 - FastPDB: Towards Bag-Probabilistic Queries at Interactive Speeds
   - Probabilistic Databases (PDBs), Bag Semantics, Expected Multiplicity, Fine-grained Complexity, Approximate Query Processing, Lineage
   - 백 시맨틱 하의 확률적 쿼리 처리는 이론적으로 PTIME에 속하지만, 정확한 계산을 위한 계보(lineage) 회로 구성에 상당한 오버헤드가 있어 비실용적.
   - 계보 회로를 미리 만들지 않고, 근사 쿼리 처리 기술을 이용해 단항식을 직접 샘플링하여 빠르고 확장 가능한 애니타임 근사 결과를 제공.

 - OBIR-tree: Secure and Efficient Oblivious Index for Spatial Keyword Queries
   - Oblivious Index, Spatial Keyword Queries, Secure Enclaves, ORAM, Intel SGX
   - 기존 공간 키워드 질의 방식은 검색, 접근, 결과량 패턴을 은닉하지 못해 보안에 취약한 문제가 있음.
   - 보안 인클레이브 기반의 은닉 인덱스 OBIR-tree를 제안하여 질의 성능을 유지하며 보안 문제를 해결함.

 - Accelerating Core Decomposition in Billion-Scale Hypergraphs
   - Hypergraphs, k-core Decomposition, Large-scale Graph Mining, Memory Efficiency
   - 기존 하이퍼그래프 코어 분해 방식은 중복 연산과 높은 메모리 소모로 인해 대규모 그래프에 확장되지 않는 문제가 있음.
   - 중복 계산 최소화 및 정점 코어 값 계산 최적화를 통해 십억 규모 하이퍼그래프도 단일 스레드로 처리 가능하도록 해결함.

 - MAST: Towards Efficient Analytical Query Processing on Point Cloud Data
   - Point Cloud, Approximate Query, Deep Models, Reinforcement Learning, Spatio-temporal Index
   - 기존 포인트 클라우드 질의 시스템은 의미 기반 분석을 지원하지 않아 비효율적인 문제가 있음.
   - 핵심 프레임 샘플링과 강화학습 기반 최적화로 딥 모델 호출을 최소화하며 질의 효율을 향상시켜 해결함.

 - 𝜆-Tune: Harnessing Large Language Models for Automated Database System Tuning
   - Database Tuning, LLMs, Configuration Optimization, Cost-based Prompting, PostgreSQL, MySQL
   - 기존 자동 튜닝은 단일 파라미터 위주로 힌트를 제공해 튜닝 범위가 제한되는 문제가 있음.
   - LLM을 활용한 프롬프트 최적화와 구성 후보 선택 전략을 통해 전체 튜닝 스크립트를 생성하여 해결함.

 - Rapid Data Ingestion through DB-OS Co-design
   - Data Ingestion, Sequential Access, DB-OS Integration, I/O Optimization, zicIO
   - 기존 시스템은 순차적 적재 시 계층 간 지연과 중복 데이터 접근으로 인한 성능 저하 문제가 있음
   - DB와 OS 간 협업으로 데이터 접근을 자동화하고 중복 페칭을 제거하여 고속 적재 문제를 해결함.
   - **흥미로운 or 신기한**

 - U-DPAP: Utility-aware Efficient Range Counting on Privacy-preserving Spatial Data Federation
   - Spatial Range Counting, Data Federation, Differential Privacy, Approximate Query Processing
   - 기존 보안 방식은 연산 비용이 높고 지연이 커서 공간 데이터 페더레이션에 실용적이지 못한 문제가 있음.
   - 차등 개인정보 보호와 근사 질의 처리를 결합하여 빠르고 정확한 질의를 가능하게 하는 U-DPAP으로 문제를 해결함.

 - Capsule: an Out-of-Core Training Mechanism for Colossal GNNs
   - Graph Neural Networks (GNN), Out-of-Core Training, GPU Memory, Scalability
   - 기존 GNN 시스템은 GPU 메모리 한계로 인해 대규모 그래프 학습에 확장성 문제가 있음
   - GPU 커널 기반의 out-of-core 학습 방식인 Capsule을 도입하여 적은 메모리로도 효율적인 대규모 GNN 학습을 가능하게 함.

 - Bursting Flow Query on Large Temporal Flow Networks
   - Temporal Flow Networks, Bursting Patterns, Max Flow, Time Interval Optimization
   - 기존 시간 흐름 네트워크에서는 시간 구간 내 폭발적인 흐름 패턴을 효율적으로 탐지하기 어려운 문제가 있음.
   - 흐름 대비 시간 구간 비율이 최대가 되도록 폭발 흐름을 찾는 BFQ 및 최적화된 BFQ* 알고리즘으로 문제를 해결함.

 - Constant Optimization Driven Database System Testing
   - Logic Bug Detection, Constant Folding, DBMS Testing, Query Correctness
   - 기존 DBMS 테스트 기법은 질의 결과의 정확성을 보장하지 못해 로직 버그를 탐지하기 어려운 문제가 있음.
   - 상수 최적화 기반의 새로운 테스팅 기법(CODDTest)을 제안하여 로직 버그를 효과적으로 탐지함.

 - A Benchmark for Data Management in Microservices
   - Microservices, Data Benchmarking, Transaction Processing, Query Processing, Constraint Enforcement
   - 기존 벤치마크는 마이크로서비스 환경에서의 데이터 관리 문제를 포괄하지 못하는 문제가 있음.
   - 마이크로서비스의 핵심 데이터 관리 과제를 측정할 수 있는 Online Marketplace 벤치마크를 통해 이 문제를 해결함.

 - Sequoia: An Accessible and Extensible Framework for Privacy-Preserving Machine Learning over Distributed Data
   - Privacy-preserving Machine Learning (PPML), Secure Computation, JAX, Distributed Training, Compiler Optimization
   - 기존 PPML 프레임워크는 보안 프로토콜과 ML 모델이 밀결합되어 접근성과 최적화가 어려운 문제가 있음.
   - Sequoia는 모델과 보안 연산을 분리하고 자동화된 컴파일러 구조를 적용해 PPML 개발성과 성능을 개선함.

 - Tribase: A Vector Data Query Engine for Reliable and Lossless Pruning Compression using Triangle Inequalities
   - Approximate Nearest Neighbor Search (ANNS), Cluster Index, Triangle Inequality, Pruning, Vector Search
   - 기존 클러스터 기반 인덱스는 세분성이 낮아 다양한 품질의 벡터와 거리 계산을 수행해야 하는 문제가 있음.
   - 다양한 거리 기준으로 클러스터를 정밀 분할하고 삼각 부등식을 활용해 손실 없이 탐색 범위를 가지치기하여 문제를 해결함.

 - SNAILS: Schema Naming Assessments for Improved LLM-Based SQL Inference
   - NL-to-SQL, Schema Naming, LLM Prompting, Natural Language Interface
   - LLM은 스키마와 질의 간 어휘 불일치로 인해 NL-to-SQL 성능에 한계가 있는 문제가 있음.
   - 자연어 친화적인 스키마 식별자를 생성하고 평가하는 벤치마크(SNAILS)를 통해 정확도를 향상시킴.

 - Dialogue Benchmark Generation from Knowledge Graphs with Cost-Effective Retrieval-Augmented LLMs
   - Dialogue Benchmark, Knowledge Graphs, Retrieval-Augmented Generation, LLMs
   - 기존 대화 벤치마크는 문서 기반 수작업 방식으로, 지식 그래프를 활용한 자동 생성이 어려운 문제가 있음.
   - Chatty-Gen은 쿼리 기반 검색과 다단계 생성 방식으로 대화 벤치마크를 자동 생성하여 정확도와 비용 문제를 해결함.

 - LCP: Enhancing Scientific Data Management with Lossy Compression for Particles
   - Particle Data, Lossy Compression, Scientific Data Management, HPC
   - 기존 손실 압축 기법은 격자 기반 데이터에만 최적화되어 입자 데이터에는 비효율적인 문제가 있음.
   - LCP는 입자 데이터 전용 공간/시간 기반 하이브리드 압축 기법을 도입해 정확도 유지와 압축 성능 문제를 해결함.

 - Disco: A Compact Index for LSM-trees
   - LSM-tree, Indexing, Read Optimization, Key-Value Store, Range Query
   - 기존 LSM-tree는 인덱스 부재로 인해 읽기 성능이 낮고 필터 의존도가 높은 문제가 있음.
   - Disco는 모든 키를 인덱싱하는 컴팩트 인덱스를 통해 I/O를 최소화하고 읽기 성능 문제를 해결함.

 - Efficiently Processing Joins and Grouped Aggregations on GPUs
   - GPU Databases, Joins, Group-By, Random Access, Query Optimization
   - 기존 GPU 기반 연산은 무작위 접근과 group-by 처리 부족으로 성능 저하 문제가 있음.
   - GFTR 기법과 최적화된 group-by 알고리즘을 통해 성능 병목을 해결하고 효율적으로 처리함.
   - **흥미로운 or 신기한**

 - HyperMR: Efficient Hypergraph-enhanced Matrix Storage on Compute-in-Memory Architecture
   - Compute-in-Memory (CIM), Matrix-Vector Multiplication, Hypergraph Partitioning, Matrix Storage
   - 기존 CIM 행렬 저장 방식은 제한된 최적화 목표와 유연성 부족으로 성능 저하 문제가 있음.
   - 하이퍼그래프 기반 저장 방식과 2단계 분할 기법을 통해 다양한 구조에 최적화된 저장 효율을 달성함.

 - InTime: Towards Performance Predictability In Byzantine Fault Tolerant Proof-of-Stake Consensus
   - BFT-PoS, MEV, Performance Predictability, Consensus Incentive Mechanism
   - 기존 BFT-PoS는 블록 제안 지연을 통한 MEV 보상 조작으로 지연 시간의 예측 가능성이 떨어지는 문제가 있음.
   - MEV 보상을 시간 기반으로 정량화하고 ARI·CTW 메커니즘을 도입해 지연을 억제함으로써 예측 가능성을 확보함.

 - LeaFi: Data Series Indexes on Steroids with Learned Filters
   - Data Series, Similarity Search, Learned Index, Pruning Optimization
   - 기존 시계열 인덱스는 가지치기 효율이 낮아 검색 연산 낭비가 큰 문제가 있음.
   - 노드 간 거리 하한을 예측하는 학습 기반 필터를 통해 가지치기를 고도화하여 검색 속도를 향상시킴.
   - **흥미로운 or 신기한**

 - PoneglyphDB: Efficient Non-interactive Zero-Knowledge Proofs for Arbitrary SQL Queries Verification
   - Zero-Knowledge Proof, SQL Query Verification, Confidentiality, PLONK, Cryptographic Circuits
   - 민감 데이터를 다루는 DB 질의에서 정확성 검증과 기밀성 보장이 동시에 어려운 문제가 있음.
   - 질의 결과에 대한 ZKP 회로를 설계하여 클라이언트가 데이터 없이도 질의 결과를 검증할 수 있도록 해결함.

 - BCviz: A Linear-Space Index for Mining and Visualizing Cohesive Bipartite Subgraphs
   - Biclique Mining, Bipartite Graphs, Graph Reduction, Indexing, Visualization
   - 기존 이중클리크 탐색은 지역 밀도만 고려하여 그래프 축소가 비효율적인 문제가 있음.
   - 연결성과 밀도를 모두 고려한 정점 순서를 통해 탐색 성능과 시각화를 동시에 향상시킴.

 - Dual-Hierarchy Labelling: Scaling Up Distance Queries on Dynamic Road Networks
   - Shortest Path, Dynamic Road Networks, Distance Query, Hierarchical Labelling
   - 기존 방법은 정적 도로망 전제를 가정하거나 동적 갱신 시 성능 저하 문제가 있음.
   - 질의와 갱신을 위한 이중 계층 구조를 사용해 동적 도로망에서도 효율적으로 거리 질의를 처리함.

 - Cardinality Estimation of LIKE Predicate Queries using Deep Learning
   - Cardinality Estimation, LIKE Predicate, N-gram, Deep Learning, Training Data Generation
   - 기존 방법은 통계 기반 접근으로 LIKE 조건에 대한 정확한 카디널리티 추정이 어려운 문제가 있음.
   - 확장 N-gram과 조건부 회귀 모델을 활용하고, 공유 가능한 쿼리 결과를 활용한 데이터 생성 기법으로 이를 해결함.

 - SHARQ: Explainability Framework for Association Rules on Relational Data
   - Association Rules, Shapley Value, Explainability, Attribute Importance, Rule Mining
   - 연관 규칙 내 요소의 상대적 중요도를 설명하기 어려운 문제가 있음.
   - 샤플리 값을 기반으로 요소의 기여도를 효율적으로 계산하는 프레임워크로 이를 해결함.

 - Randomized Sketches for Quantile in LSM-tree based Store
   - Quantile Estimation, LSM-tree, Sketching, KLL, Randomized Algorithm
   - 기존 스트리밍 기반 분위수 추정은 LSM-tree 구조에서 선형 I/O 비용 문제를 가짐.
   - 사전 계산된 랜덤화 스케치 기법으로 서브선형 I/O 비용을 달성하여 이를 해결함.

 - Centrum: Escape from the Gaussian Process World! Enhancing Database Auto-tuning with Tree-Ensemble Bayesian Optimization(Centrum: Model-based Database Auto-tuning with Minimal Distributional Assumptions)
   - Auto-tuning, Bayesian Optimization, Gradient Boosting, Conformal Prediction, Distribution-free
   - GP 기반 튜너는 분포 가정이 자주 위배되어 성능이 떨어지는 문제가 있음
   - gradient boosting과 conformal prediction을 결합한 Centrum으로 분포 가정 없는 효율적인 자동 튜닝을 실현함.

 - Revisiting the Design of In-Memory Dynamic Graph Storage
   - Dynamic Graph Storage, Real-time Analytics, Concurrency Control, Memory Overhead
   - 기존 동적 그래프 저장소는 공간 오버헤드와 동시성 문제로 성능 저하 문제가 있음.
   - 공통 추상화 기반 테스트로 병목을 분석하고 개선 방향을 제시하여 문제를 해결함.

 - H-Rocks CPU-GPU accelerated RocksDB on Persistent Memory
   - RocksDB, Persistent Memory, GPU Acceleration, Key-Value Store
   - 기존 RocksDB 기반 저장소는 CPU에만 의존하여 처리 성능이 낮은 문제가 있음.
   - GPU 병렬성과 고대역폭 메모리를 활용해 RocksDB 연산을 가속하여 문제를 해결함.

 - Data-Centric Machine Learning Pipeline Orchestration
   - Machine Learning Pipelines, Data Selection, Triggering Policies, Continuous Training
   - 실제 ML 파이프라인은 데이터가 지속적으로 증가해 전체 재학습이 비효율적인 문제가 있음.
   - 선택적 데이터 및 트리거링 정책을 설정하고 오케스트레이션하여 지속적 학습을 효율화함.

 - DiskGNN: Bridging I/O Efficiency and Model Accuracy for Out-of-Core GNN Training
   - Out-of-Core GNN, Graph Sampling, I/O Efficiency, Feature Packing, Pipelined Training
   - 기존 시스템은 디스크 기반 학습에서 read amplification 문제 또는 정확도 저하 문제가 있음.
   - 오프라인 샘플링과 연속적 특성 배치, 계층적 캐시, 파이프라인 학습으로 효율과 정확도를 동시에 해결함.

 - Federated Heavy Hitter Analytics with Local Differential Privacy
   - Federated Analytics, Local Differential Privacy (LDP), Heavy Hitters, Privacy-Preserving, Prefix Tree
   - 기존 LDP 기반 연합 분석은 노이즈로 인해 유용성이 낮고 통신/계산 비용이 큰 문제가 있음.
   - 접두 트리 기반 구조와 적응형/합의 기반 전략으로 정확도와 효율성 문제를 해결함.

 - DISCES: Systematic Discovery of Event Stream Queries
   - Event Stream, Query Discovery, Complex Event Processing, Algorithm Design Space
   - 기존 이벤트 질의 발견 방식은 설계 기준이 불명확하고 데이터베이스 적합성을 판단하기 어려운 문제가 있음
   - 질의 발견 알고리즘의 설계 공간을 정의하고 4가지 실행 민감도별 알고리즘으로 이를 해결함.

 - Pandora: An Efficient and Rapid Solution for Persistence-Based Tasks in High-Speed Data Streams
   - Persistence, Data Streams, Approximate Data Structures, SIMD, Anomaly Detection
   - 기존 지속성 기반 기법은 해시 충돌과 메모리 제약으로 정확도와 처리 속도에 문제가 있음.
   - 비지속 항목을 우선 제거하고 SIMD 기반으로 최적화된 Pandora 구조로 이 문제를 해결함.

 - Computing Approximate Graph Edit Distance via Optimal Transport
   - Graph Edit Distance, Optimal Transport, Inverse Sinkhorn, Gromov-Wasserstein, Graph Similarity
   - 기존 GED 근사 기법은 전체 그래프 맥락을 고려하지 못해 정확도에 한계가 있음.
   - 최적 수송 기반의 지도/비지도 기법과 그 앙상블로 GED 계산 정확도와 일반성을 개선함.

 - B-Trees Are Back: Engineering Fast and Pageable Node Layouts
   - B-Tree, Hybrid Storage, Pageable Node, Variable-Length Records, Adaptive Layout
   - 기존 B-Tree 연구는 고정 길이 레코드만을 가정해 실제 시스템에 적용이 어려운 문제가 있음.
   - 가변 길이 레코드를 지원하고 성능 최적화된 적응형 레이아웃을 적용하여 인메모리 구조와도 경쟁할 수 있도록 개선함.

 - Largest Triangle Sampling for Visualizing Time Series in Database
   - Time Series Visualization, Sampling, Largest Triangle, Convex Hull, Iterative Refinement
   - 기존 시계열 샘플링 기법은 최적성이 부족하고 쿼리 효율이 낮은 문제가 있음.
   - Convex Hull 기반 반복 정제를 통해 시각 품질을 개선하고 효율적인 삼각형 샘플링을 실현함.

 - ISSD: Indicator Selection for Time Series State Detection
   - Time Series, State Detection, Indicator Selection, Multi-objective Optimization
   - 기존 시계열 상태 탐지 기법은 지표 선택을 수동으로 전제하여 확장성과 효율성이 부족한 문제가 있음.
   - 상태 정보 보존 기준으로 지표를 자동 선택하는 최적화 기반 ISSD 기법을 통해 이 문제를 해결함.

 - RLER-TTE: An Efficient and Effective Framework for En Route Travel Time Estimation with Reinforcement Learning
   - Travel Time Estimation, Reinforcement Learning, Real-time Prediction, Decision Maker, Spatio-temporal Modeling
   - 기존 ER-TTE 방법은 실제 교통의 복잡성과 동적 특성을 반영하지 못해 실시간 예측 정확도와 효율성이 낮은 문제가 있음.
   - 강화학습 기반 의사결정과 어텐션 기반 시공간 인코딩으로 효율성과 정확도를 높이는 새로운 예측 프레임워크를 제안하여 해결함.

 - An Adaptive Benchmark for Modeling User Exploration of Large Datasets
   - User Exploration, DBMS Benchmarking, SQL Simulation, Dashboard Interaction, Performance Evaluation
   - 기존 벤치마크는 사용자의 탐색 흐름과 분석 목표 변화를 정확히 반영하지 못해 DBMS 성능 병목을 놓치는 문제가 있음.
   - 사용자 인터랙션과 목표 쿼리를 기반으로 탐색 흐름을 시뮬레이션하는 SIMBA 벤치마크를 통해 이 문제를 해결함.

 - Reliable Text-to-SQL with Adaptive Abstention
   - Text-to-SQL, Schema Linking, Human-in-the-loop, Conformal Prediction, Query Reliability
   - 기존 Text-to-SQL 시스템은 문맥 부족이나 애매한 질의에서 신뢰할 수 없는 SQL을 생성하는 문제가 있음.
   - 스키마 연결 단계에서 오류를 감지하고 필요시 질의를 중단하거나 사용자 개입을 요청하는 RTS 프레임워크로 이 문제를 해결함.

 - Entity/Relationship Graphs (Entity/Relationship Graphs: Principled Design, Modeling, and Data Integrity Management of Graph Databases)
   - E/R Modeling, Property Graphs, Referential Integrity, Data Redundancy, PG-Schema
   - 기존 그래프 데이터베이스는 무결성 보장과 중복 제거가 어려운 문제가 있음.
   - E/R 키와 링크를 활용한 원칙 기반 E/R 그래프 모델로 이 문제를 해결함.

 - Cohesiveness-aware Hierarchical Compressed Index for Community Search on Attributed Graphs
   - Community Search, Attributed Graphs, Indexing, Cohesiveness, Graph Compression
   - 기존 커뮤니티 탐색 기법은 느리고 일반화가 어려운 문제가 있음.
   - 응집성 인지 계층 압축 인덱스를 통해 효율성과 일반화를 동시에 해결함.

 - Minimum Spanning Tree Maintenance in Dynamic Graphs
   - Minimum Spanning Tree, Dynamic Graphs, Edge Replacement, Graph Maintenance
   - 기존 MST 유지 기법은 이론 중심이고 실질적인 효율성이 부족한 문제가 있음.
   - 각 트리 엣지에 대체 엣지를 유지하여 즉각적인 갱신이 가능하도록 하여 이 문제를 해결함.

 - Data Chunk Compaction in Vectorized Execution
   - Vectorized Execution, Data Chunk, Compaction, Hash Join, Runtime Optimization
   - 해시 조인 등으로 인해 벡터화 실행에서 작은 청크가 누적되어 성능 저하 문제가 있음.
   - 학습 기반 임계값 조정과 논리적 압축 기법으로 청크 압축을 최적화하여 이 문제를 해결함.

 - Progressive entity resolution: a design space exploration
   - Entity Resolution, Progressive Algorithms, Record Linkage, Deduplication, Matching Pipeline
   - 기존 ER 기법은 일괄 처리 방식으로 실시간 응용에 부적합한 문제가 있음.
   - 필터링, 가중치, 스케줄링, 매칭으로 구성된 파이프라인 기반 점진적 ER 프레임워크로 이 문제를 해결함.

 - Parallel kd-tree with Batch Updates
   - Parallel kd-tree, Batch Updates, Cache Efficiency, kNN Query, Range Query
   - 기존 병렬 kd-tree는 트리 생성과 갱신에서 병목과 성능 저하 문제가 있음.
   - 병렬 생성과 일괄 갱신을 지원하는 Pkd-tree로 이 문제를 해결함.

 - User-Centric Property Graph Repairs
   - Property Graph, Denial Constraints, Graph Repair, Interactive Systems
   - 실제 사용자 기반의 그래프는 무결성 위반 문제로 인한 수정 필요성이 있는 문제가 있음.
   - 사용자 참여 기반의 인터랙티브 수선 알고리즘으로 이 문제를 해결함.

 - In-Database Time Series Clustering
   - Time Series Clustering, In-Database Processing, K-Shape, IoT, Apache IoTDB
   - IoT 시나리오에서는 대용량 시계열 데이터를 기존 클러스터링 방식으로 처리하기 어려운 문제가 있음.
   - K-Shape 및 Medoid-Shape의 in-database 클러스터링 기법으로 이 문제를 해결함.

 - Boosting OLTP Performance with Per-Page Logging on NVDIMM
   - OLTP, NVDIMM, Logging, Redo-less Recovery, Multi-Versioning
   - SSD 기반 OLTP에서 로그 내구성 문제로 인해 성능이 심각하게 제한되는 문제가 있음.
   - NVDIMM에 페이지 단위 로그를 저장하여 내구성 문제를 해결하고 복구 및 다중 버전 성능을 향상함.

 - Shapley Value Estimation based on Differential Matrix
   - Shapley Value, Cooperative Game Theory, Monte Carlo, Differential Matrix, Variance Reduction
   - 기존 Monte Carlo 방식은 직접 추정으로 인한 높은 분산으로 정확한 Shapley 값 계산이 어려운 문제가 있음.
   - Shapley 값의 차이를 기반으로 하는 differential matrix를 추정하고 최소자승 최적화로 값을 복원하여 이 문제를 해결함.

 - SecureXGB: A Secure and Efficient Multi-party Protocol for Vertical Federated XGBoost
   - Vertical Federated Learning, XGBoost, Secret Sharing, Secure Multi-party Computation, Privacy-preserving Machine Learning
   - 기존 수직 연합 XGBoost 프로토콜은 많은 비선형 연산으로 인해 실행 효율에 문제가 있음.
   - 병렬 샘플 셔플링, 선형 점수 계산, 최소 비교 기반 분할 선택으로 이 문제를 해결함.

 - On Graph Representation for Attributed Hypergraph Clustering
   - Hypergraph Clustering, Node Attributes, Cluster-Number-Free, Graph Representation, Modularity Optimization
   - 기존 AHC 방법은 행렬 분해 기반으로 계산 비용이 높고 군집 수 추정이 필요해 정확도 저하 문제가 있음.
   - 군집 수 추정 없이 그래프 표현 기반으로 속성 및 구조 통합하여 클러스터링 품질과 속도를 개선함.

 - Nezha: An Efficient Distributed Graph Processing System on Heterogeneous Hardware
   - Distributed Graph Processing, Heterogeneous Hardware, RDMA, CPU-GPU Cooperation, Workload Balancing
   - 기존 분산 그래프 처리 시스템은 통신 비용이 높고 각 머신의 계산 자원을 충분히 활용하지 못하는 문제가 있음.
   - RDMA 기반 통신 최적화 및 CPU-GPU 협력 실행으로 계산 자원 활용도와 처리 효율을 개선함.

 - Practical DB-OS Co-Design with Privileged Kernel Bypass
   - Database-OS Co-Design, Virtualization, Kernel Bypass, Snapshotting, In-Kernel Buffer Pool
   - 기존 OS 인터페이스는 DB 워크로드 요구를 충족하지 못하고 설계 제약과 보안 문제가 있음.
   - 가상화 기반 권한 상승 방식으로 DB에 최적화된 커널 기능을 제공하여 성능 및 호환성 문제를 해결함.

 - QURE: AI-Assisted and Automatically Verified UDF Inlining
   - UDF Optimization, SQL Translation, Large Language Models, Formal Verification, Query Performance
   - 기존 UDF 최적화 기법은 패턴 기반으로 범용성이 낮고 새로운 구조에 취약한 문제가 있음.
   - LLM 기반 SQL 변환과 자동 형식 검증을 통해 고성능과 확장성을 동시에 확보하여 이 문제를 해결함.

 - A Rank-Based Approach to Recommender System's Top-K Queries with Uncertain Scores
   - Top-K Queries, Recommender Systems, Probabilistic Ranking, Uncertain Scores, RankDist Algorithm
   - 추천 시스템에서 점수가 불확실해 정확한 Top-K 순위를 계산하기 어려운 문제가 있음.
   - 확률적 랭킹 기반의 RankDist 알고리즘을 통해 불확실성을 반영한 추천 순위를 최적으로 계산하여 해결함.

 - VEGA: An Active-tuning Learned Index with Group-Wise Learning Granularity
   - Learned Index, Model Granularity, Key Repositioning, Online Training, Lookup Optimization
   - 기존 learned index는 정확도와 성능을 동시에 만족하지 못하는 문제가 있음.
   - 키 그룹화와 위치 조정을 통해 정확도와 성능을 동시에 만족하는 VEGA 인덱스를 제안하여 해결함.

 - AquaPipe: A Quality-Aware Pipeline for Knowledge Retrieval and Large Language Models
   - Retrieval-Augmented Generation (RAG), Approximate Nearest Neighbor Search (ANNS), Latency Overlap, Prefetching, Pipelining
   - 디스크 기반 지식 검색이 느려 RAG 시스템의 응답 시간이 증가하는 문제가 있음.
   - 검색과 추론을 파이프라인으로 겹쳐 지연을 줄이고 정확도를 유지하는 AquaPipe를 제안하여 해결함.

 - TGraph: A Tensor-centric Graph Processing Framework
   - Graph Processing, Tensor Computation Runtime (TCR), Hardware Acceleration, XPU, Deep Learning Frameworks
   - 하드웨어 특화 그래프 시스템은 이식성이 낮아 다양한 가속기에서 활용하기 어려운 문제가 있음.
   - 텐서 기반 계산 모델을 이용해 다양한 DL 프레임워크 및 가속기에서 실행 가능한 범용 그래프 처리 프레임워크로 해결함.

 - Efficiently Counting Triangles in Large Temporal Graphs
   - Temporal Graphs, Triangle Counting, δ-temporal Triangle, Indexing, Query Time Window
   - 시계열 그래프에서 삼각형을 효율적으로 세기 위한 기존 방식은 느리고 확장성이 떨어지는 문제가 있음.
   - δ-temporal triangle 모델 기반의 온라인 및 인덱스 기반 알고리즘을 제안하여 해당 문제를 해결함.

 - Multi-Level Graph Representation Learning Through Predictive Community-based Partitioning
   - Graph Representation Learning (GRL), Community Detection, Predictive Partitioning, Parallel Processing
   - 기존 GRL은 커뮤니티 분할 방식 선택이 정적이며 효율성과 정확도 모두에서 한계가 있는 문제가 있음.
   - 그래프 기반 예측 모델을 통해 최적의 분할 방식을 선택하고 병렬 처리를 통해 정확도와 효율성을 동시에 개선함.

 - Efficient Maximum s-Bundle Search via Local Vertex Connectivity
   - s-Bundle, Cohesive Subgraph, Branch-and-Bound, Local Vertex Connectivity, Graph Search
   - 기존 s-bundle 탐색은 가지치기 중심으로 최악의 시간 복잡도 O*(2ⁿ) 문제를 가짐.
   - 정점 연결성 기반 정렬과 Symmetric-BK 전략을 통해 시간 복잡도를 개선하고 성능을 획기적으로 향상함.

 - Automatic Database Configuration Debugging using Retrieval-Augmented Language Models
   - Database Configuration, Debugging, Large Language Models, Retrieval-Augmented Generation, Telemetry
   - 기존 DBMS 설정 디버깅은 복잡성과 전문성 요구로 인해 자동화가 어려운 문제가 있음.
   - LLM과 도메인 문서 검색을 결합한 RAG 기반 시스템으로 설정 디버깅 자동화를 효과적으로 해결함.

 - A Local Search Approach to Efficient (k,p)-Core Maintenance
   - (k,p)-Core, Core Maintenance, Local Search, Dynamic Graphs, p-number
   - 기존 방식은 p-number 갱신 시 범위 내 모든 정점을 다시 계산해야 하여 비효율적인 문제가 있음.
   - 엣지 기준 국소 탐색을 통해 영향을 받는 정점을 선별적으로 갱신함으로써 성능을 크게 향상시킴.

 - B$\circledS X$ : Subgraph Matching with Batch Backtracking Search
   - Subgraph Matching, Backtracking, Batch Processing, Search Box, EPS
   - 기존 서브그래프 매칭은 정점 하나씩 탐색하여 중복 계산이 많은 문제가 있음.
   - 비슷한 탐색 공간을 묶어 일괄 처리하고 조기 중단 및 병렬 임베딩으로 속도를 향상시킴.

 - DEG: Efficient Hybrid Vector Search Using the Dynamic Edge Navigation Graph
   - Hybrid Vector Query (HVQ), Approximate Nearest Neighbor Search (ANNS), Bimodal Data, Pareto Frontier, Alpha-weighted Similarity
   - 기존 HVQ 인덱스는 쿼리 가중치 α 값 변화에 따라 성능 저하 문제가 있음.
   - 다양한 α 값에서도 효율과 정확성을 유지하는 그래프 기반 인덱스 DEG를 제안하여 해결함.

 - Efficient Index Maintenance for Effective Resistance Computation on Evolving Graphs
   - Effective Resistance, Evolving Graphs, Random Walks, Loop-Erased Walks, Index Maintenance
   - 기존 행렬 기반 인덱스는 업데이트 처리와 저장 효율에서 한계가 있는 문제가 있음.
   - 무작위 워크 샘플 기반 인덱스와 사이클 단위 루프 제거 기법으로 빠른 업데이트와 효율적 계산을 가능하게 하여 문제를 해결함.

 - Schema-Based Query Optimisation for Graph Databases
   - Recursive Graph Queries, Graph Schema, Type Inference, Query Optimisation, Soundness and Completeness
   - 재귀 그래프 질의 성능이 낮고 구조적 제약을 활용하지 못하는 문제가 있음.
   - 스키마 기반 타입 추론으로 질의에 구조 정보를 주입하여 성능을 향상시키는 방식으로 해결함.

 - SymphonyQG: towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search
   - Approximate Nearest Neighbor (ANN), Quantization, Graph-based Index, SIMD, FastScan
   - 그래프 기반 ANN 탐색은 메모리 접근 병목과 거리 계산 비용으로 인해 성능 저하 문제가 있음.
   - 양자화와 그래프를 효과적으로 통합하여 탐색 성능을 향상시키는 SymphonyQG 기법으로 해결함.

 - Density Decomposition of Bipartite Graphs
   - Bipartite Graph, Dense Subgraph, (α,β)-Density, Flow Algorithm, Decomposition
   - 기존 이분 그래프 조밀 모델은 계산 복잡도나 밀도 포착의 한계로 인해 활용이 어려운 문제가 있음.
   - (α,β)-dense 모델을 기반으로 밀도 분해 계층을 구성하고 네트워크 플로우 기반 알고리즘으로 해결함.

 - An Elephant Under the Microscope: Analyzing the Interaction of Optimizer Components in PostgreSQL
   - Query Optimizer, PostgreSQL, Cardinality Estimation, Cost Model, Plan Generation 
   - 질의 최적화기의 구성 요소 간 상호작용이 충분히 이해되지 않아 예상 외의 결과나 연구 낭비 문제가 있음.
   - PostgreSQL 기반 실험 분석을 통해 구성 요소 간 영향을 규명하고, 개선 방향을 제시함.
   - **흥미로운 or 신기한**

 - MEMO: Fine-grained Tensor Management For Ultra-long Context LLM Training
   - LLM Training, Memory Management, FlashAttention, GPU Fragmentation, MFU
   - 긴 컨텍스트 학습 시 GPU 메모리 소모 및 조각화로 인해 학습 효율이 낮아지는 문제가 있음.
   - CPU로의 세밀한 활성값 오프로딩 및 메모리 최적화 기법으로 메모리 조각화와 재계산 문제를 해결함.

 - SPAS: Continuous Release of Data Streams under w-Event Differential Privacy
   - Data Streams, Differential Privacy, Sliding Window, Adaptive Publishing, Sparse Vector
   - 기존 스트림 공개 기법은 휴리스틱 기반으로 데이터에 맞지 않아 효과가 낮은 문제가 있음.
   - 데이터 기반 최적 공개 전략 예측과 가중 희소 벡터 기법으로 적응성과 정확도를 확보함.

 - Aster: Enhancing LSM-structures for Scalable Graph Database
   - Graph Database, LSM-tree, Edge Updates, Gremlin, Key-Value Store
   - 기존 그래프 DB는 빈번한 업데이트와 대규모 그래프 처리에서 성능 저하 문제가 있음.
   - 그래프에 최적화된 LSM 구조와 적응형 엣지 처리 기법으로 대규모 환경에서 성능을 향상시킴.

 - An experimental comparison of tree-data structures for connectivity queries on fully-dynamic undirected graphs
   - Fully-Dynamic Graphs, Connectivity Queries, Tree Structures, Experimental Evaluation
   - 기존 트리 기반 연결성 질의 구조는 실제 환경에서 공간, 시간, 유지비용 문제로 인해 사용에 어려움이 있음.
   - 다양한 구현과 실험을 통해 문제점을 분석하고 실사용 가능한 연결성 질의 구조 구현 방안을 제안함.

 - CRDV: Conflict-free Replicated Data Views
   - CRDTs, SQL Views, Merge Semantics, Query Optimization, Replicated Tables
   - 기존 CRDT 기반 SQL 시스템은 병합 의미론 정의 및 쿼리 최적화에 제약이 있는 문제가 있음.
   - SQL 뷰 기반 계층으로 병합 의미론과 쿼리 최적화를 통합하여 성능과 확장성을 동시에 확보함.

 - Graph-Based Vector Search: An Experimental Evaluation of the State-of-the-Art
   - Vector Search, Graph-based Methods, Incremental Insertion, Neighborhood Diversification, Scalability
   - 벡터 데이터 분석에서 그래프 기반 검색 방법의 설계 방식과 성능 차이에 대한 정량적 평가가 부족한 문제가 있음.
   - 다양한 패러다임에 기반한 최신 기법들을 실험적으로 비교하여 강점과 한계를 규명하고 확장성 관점에서의 설계 인사이트를 제시함.

 - Subspace Collision: An Efficient and Accurate Framework for High-dimensional Approximate Nearest Neighbor Search
   - Approximate Nearest Neighbor (ANN), High-dimensional Search, Subspace Collision, Clustering-based Index
   - 기존 ANN 기법은 정확도와 속도, 이론적 보장을 동시에 만족시키기 어려운 문제가 있음.
   - 파레토 기반 거리 척도와 경량 인덱스를 활용한 SuCo 시스템으로 이 문제를 해결함.

 - Ultraverse: A System-Centric Framework for Efficient What-If Analysis for Database-Intensive Web Applications
   - What-if Analysis, Symbolic Execution, Query Dependency, Web Applications, SQL Procedure
   - 기존 What-if 분석은 애플리케이션과 데이터베이스를 동시에 고려하지 못해 분석 효율이 떨어지는 문제가 있음.
   - 동적 심볼릭 실행과 쿼리 의존성 분석을 통해 트랜잭션을 최적화하여 분석 속도를 크게 향상함.

 - Deep Overlapping Community Search via Subspace Embedding
   - Overlapping Community Search (OCS), Personalized Search, Sparse Subspace Filter (SSF), Multi-hop Attention
   - 기존 오버래핑 커뮤니티 탐색은 사용자 맞춤형 탐색을 지원하지 못하는 문제가 있음.
   - SSF 프레임워크와 SMN 모델을 통

 - DataVinci: Learning Syntactic and Semantic String Repairs
   - String Data Cleaning, Error Detection, Regex Pattern Learning, LLM-based Repair
   - 기존 문자열 정제 기법은 오류 탐지 또는 사용자 예제 기반 수정에만 의존하는 문제가 있음.
   - 정규식 기반 다수 패턴 학습과 LLM 추상화를 결합한 자동 오류 탐지 및 수정 방식으로 문제를 해결함.

 - Optimizing Block Skipping for High-Dimensional Data with Learned Adaptive Curve
   - Block Skipping, High-Dimensional Data, SMA, Adaptive Projection, Machine Learning
   - 기존 SMA는 고차원 테이블에서 데이터 레이아웃에 따라 성능 저하 문제가 있음.
   - 적응형 곡선을 학습하여 데이터 특성에 맞는 레이아웃을 생성함으로써 블록 스키핑 문제를 해결함.
 
 - Online Detection of Anomalies in Temporal Knowledge Graphs with Interpretability
   - Temporal Knowledge Graphs, Anomaly Detection, Rule Graph, Interpretability, Online Learning
   - 기존 이상 탐지 기법은 시간 지식 그래프의 의미 정보 반영과 해석 가능성이 부족한 문제가 있음.
   - TKG를 규칙 그래프로 요약하여 해석 가능한 이상 탐지를 수행함으로써 이 문제를 해결함.

 - Camel: Efficient Compression of Floating-Point Time Series
   - Time Series Compression, Floating-Point, XOR Alternative, Integer-Decimal Separation, Indexable Compression
   - 기존 압축 기법은 스트리밍 부적합성과 낮은 압축률 문제를 동시에 갖는 문제가 있음.
   - 정수/소수 분리 압축 및 고압축 가능 값 선택으로 압축률과 효율성을 동시에 해결함.

 - LSMGraph: A High-Performance Dynamic Graph Storage System with Multi-level CSR
   - Dynamic Graph Storage, LSM-tree, CSR, Read/Write Optimization, Version Control
   - 기존 시스템은 읽기/쓰기 증폭 문제로 인해 성능을 동시에 최적화하기 어려운 문제가 있음.
   - LSM-tree와 CSR 구조를 결합하고 정점 버전 관리를 통해 읽기/쓰기 성능을 동시에 해결함.

 - Navigating Labels and Vectors: A Unified Approach to Filtered Approximate Nearest Neighbor Search
   - Filtered ANNS, Label Navigating Graph, High-Dimensional Vectors, Vector Search, Structured Filtering
   - 벡터 유사도와 레이블 조건을 동시에 처리하는 필터링 ANNS에서 기존 방식은 성능과 정확성 모두에 한계가 있는 문제가 있음.
   - 레이블 포함 관계를 그래프로 모델링하여 필터 조건에 맞는 벡터만 정확히 탐색함으로써 이 문제를 해결함.

 - Buffered Persistence in B+ Trees
   - Non-volatile Memory, B+ Tree, Crash Consistency, Epoch-based Logging, Delayed Persistence
   - NVM 기반 B+ 트리는 높은 성능이 가능하지만, 장애 일관성을 위한 플러시 명령이 성능 저하를 유발하는 문제가 있음.
   - 몇 밀리초 단위의 epoch 내 지연 저장을 통해 캐시 재사용을 늘리고 쓰기 비용을 줄여 이 문제를 해결함.

 - GOLAP: A GPU-in-Data-Path Architecture for High-Speed OLAP
   - GPU Acceleration, OLAP, Compressed Data Scan, High Bandwidth, SSD Streaming
   - 기존 OLAP 시스템은 SSD 기반에서 메모리급 대역폭을 달성하기 어려운 문제가 있음.
   - SSD에서 압축 데이터를 GPU로 직접 스트리밍하여 실시간 압축 해제 및 프루닝 기법으로 이 문제를 해결함.

 - DPconv: Super-Polynomially Faster Join Ordering
   - Join Ordering, Query Optimization, Subset Convolution, Dynamic Programming, Cost Functions
   - 기존 조인 순서 결정 알고리즘은 O(3ⁿ) 복잡도로 대규모 쿼리에서 실행 시간이 과도한 문제가 있음.
   - 부분집합 컨볼루션 기반의 DPconv 프레임워크로 해당 복잡도를 극복하여 실행 시간을 대폭 줄이는 방식으로 해결함.
   - **흥미로운 or 신기한**

 - iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search
   - Range Filtering, Approximate Nearest Neighbor, Graph Indexing, Elemental Graphs, High-dimensional Data
   - 기존 방식은 모든 수치 범위에 대해 전용 인덱스를 압축 저장하지만 손실 압축으로 인해 성능 저하 문제가 있음.
   - 쿼리 시점에 적은 수의 사전 구축된 그래프를 조합해 동적으로 인덱스를 구성하는 방식으로 문제를 해결함.

 - CtxPipe: Context-aware Data Preparation Pipeline Construction for Machine Learning
   - Data Preparation, Feature Engineering, AutoML, Reinforcement Learning, Context-aware Pipeline
   - 기존 자동 파이프라인 구축 방식은 특징 품질이 낮아 모델 정확도가 떨어지고 실행 시간이 오래 걸리는 문제가 있음.
   - 데이터 의미를 파악해 구성요소 선택을 가이드하는 방식으로 모델 성능과 효율성 문제를 동시에 해결함.

 - Multivariate Time Series Cleaning under Speed Constraints
   - Time Series, Multivariate Cleaning, Data Repair, Speed Constraint, Online Algorithm
   - 기존 시계열 정제는 단변량 기반이라 다변량 상관관계를 활용하지 못해 정확도가 떨어지는 문제가 있음.
   - 데이터 분포를 크게 바꾸지 않으면서 최소한의 수정으로 정제하는 방법을 제시해 정확도와 속도 문제를 해결함.

 - Finding Logic Bugs in Spatial Database Engines via Affine Equivalent Inputs
   - Spatial DBMS, Logic Bugs, Geometry-aware SQL, Affine Transformation, Automated Testing
   - 공간 DBMS에서 논리 오류는 정확한 결과 검증 기준이 없어 탐지가 어려운 문제가 있음.
   - 동일한 기하학적 의미를 갖는 입력 집합(AEI)을 활용한 결과 비교 기법과 자동 SQL 생성기로 논리 오류 탐지를 가능하게 함.

 - A Universal Sketch for Estimating Heavy Hitters and Per-Element Frequency Moments in Data Streams with Bounded Deletions
   - Data Streams, Turnstile Model, Heavy Hitters, Frequency Moments, Online Estimation
   - 기존 삽입 전용 모델만을 고려해 삭제가 있는 데이터 스트림에서 통계 추정이 어려운 문제가 있음.
   - 삭제를 허용하는 스케치 구조와 온라인 추정 기법을 통해 다양한 지표를 동시에 추정하도록 해결함.

 - Pasta: A Cost-Based Optimizer for Generating Pipelining Schedules for Dataflow DAGs
   - DAG Workflow, Pipelined Execution, Cost-based Optimization, Scheduling
   - 기존 워크플로우 시스템은 파이프라인 스케줄링을 단순 휴리스틱으로 처리해 최적화가 어려운 문제가 있음.
   - 연산자 특성과 비용을 고려한 최적화 기법(Pasta)을 통해 고품질 파이프라인 실행 계획을 생성하도록 해결함.

 - Towards a Converged Relational-Graph Optimization Framework
   - SQL/PGQ, SPJM, Relational-Graph Query Optimization, DuckDB
   - 기존 SQL/PGQ 질의는 그래프 특성을 반영하지 못해 최적화에 한계가 있는 문제가 있음.
   - 관계형 및 그래프 최적화를 통합한 프레임워크(RelGo)를 통해 그래프 질의도 효과적으로 최적화함.

 - Disclosure-compliant Query Answering
   - Data Disclosure, Access Control, Data Masking, Query Rewriting, Information Utility
   - 기존 시스템은 민감한 정보 보호를 위한 정책 준수 및 부분 공개를 동시에 지원하지 못하는 문제가 있음.
   - 사용자 질의를 마스킹된 데이터에 맞게 변환하고 정보 손실이 최소화되도록 선택하는 Mascara 시스템으로 해결함.

 - Memento Filter: A Fast, Dynamic, and Robust Range Filter
   - Range Filter, False Positives, Dynamic Filter, B-Tree, Key-Value Store
   - 기존 Range Filter는 동적 데이터셋을 지원하지 못하고 상관된 질의에서 허위 긍정률이 높아지는 문제가 있음.
   - 키 클러스터링 기반으로 동적 삽입·삭제 및 확장을 지원하는 Memento Filter로 문제를 해결함.

 - GEIL: A Graph-Enhanced Interpretable Data Cleaning Framework with Large Language Models (GIDCL: A Graph-Enhanced Interpretable Data Cleaning Framework with Large Language Models)
   - Data Cleaning, Interpretable AI, Graph Neural Network, Large Language Models, Few-shot Learning
   - 기존 데이터 정제는 규칙 의존성과 복잡 오류 패턴 처리 미흡 문제가 있음.
   - GNN과 LLM을 활용하여 해석 가능성과 복잡 오류 정정이 가능한 GIDCL로 문제를 해결함

 - Live Patching for Distributed In-Memory Key-Value Stores
   - Live Patching, Redis Cluster, Rolling Update, High Availability, In-Memory Database
   - 기존 롤링 업데이트는 메모리 상태 복원·동기화로 인해 보안 패치가 지연되는 문제가 있음.
   - 노드 재시작 없이 메모리 내 직접 패치하는 경량 라이브 패칭으로 문제를 해결함.
   - **흥미로운 or 신기한**

 - Provenance-Enabled Explainable AI
   - Explainable AI (XAI), Provenance Graph, Black-Box Model, Interpretability, Efficiency
   - 기존 XAI는 모델과 밀결합되고 중복 연산으로 비효율적인 문제가 있음.
   - 프로비넌스 그래프로 연산을 분리하고 불필요한 요소를 제거하여 문제를 해결함.

 - Constant-time Connectivity Querying in Dynamic Graphs
   - Connectivity Query, Dynamic Graph, Disjoint Set, Spanning Tree, Constant-Time
   - 기존 연결성 질의 기법은 자주 변경되는 그래프에서 성능이 저하되는 문제가 있음.
   - 분리 집합 트리와 스패닝 트리를 결합하여 질의 성능과 업데이트 성능을 동시에 해결함.

 - Vectorizing Distributed Graph Computations made Automated (Automating Vectorized Distributed Graph Computation)
   - Multi-instance Algorithms, Graph Computation, AutoMI, SIMD, Vectorization
   - 다중 인스턴스 그래프 알고리즘을 수작업으로 작성하는 것은 어렵고 오류 가능성이 있는 문제가 있음.
   - 정점 중심 알고리즘을 자동으로 벡터화된 다중 인스턴스 형태로 변환하는 AutoMI 프레임워크로 이 문제를 해결함.

 - High-Performance Query Processing with NVMe Arrays: Spilling without Killing Performance
   - Query Processing, NVMe, Adaptive Materialization, Compression, Out-of-Memory
   - 기존 시스템은 인메모리 연산과 외부메모리 연산 간 성능 선택 문제가 있음.
   - 인메모리 성능을 유지하면서도 NVMe 기반 스필링을 효율적으로 처리하는 adaptive materialization과 self-regulating compression으로 이 문제를 해결함.

 - Personalized Truncation for Personalized Privacy
   - Differential Privacy, Personalized DP, Sum Estimation, Foreign-key Constraints
   - 기존 PDP 기법은 다양한 개인정보 요구를 반영하나 기본 연산에서 낮은 유틸리티 문제가 있음.
   - 각 사용자에 맞는 truncation을 적용하는 새로운 메커니즘으로 정확도를 높여 이 문제를 해결함.

 - Understanding and Reusing Test Suites Across Database Systems
   - Database Testing, Test Suite Reuse, DBMS Compatibility, SQuaLity
   - DBMS 간 테스트 스위트가 포맷과 종속성 차이로 인해 재사용이 어려운 문제가 있음.
   - 다양한 DBMS 테스트를 통합한 SQuaLity를 통해 재사용 가능성과 그 효과를 실증적으로 입증함.

 - Common Neighborhood Estimation over Bipartite Graphs under Local Differential Privacy
   - Bipartite Graph, Common Neighbor, Local Differential Privacy, Edge Privacy, Estimation
   - 이분 그래프에서 공통 이웃 추정은 개인정보 노출로 인해 로컬 프라이버시 하에서 수행이 어려운 문제가 있음.
   - 다단계 후보 필터링과 추정기 통합, 예산 최적화를 통해 edge LDP 하에서도 효율적이고 정확한 추정이 가능함.

 - An Efficient and Exact Algorithm for Locally h-Clique Densest Subgraph Discovery
   - Community Search, h-Clique, Densest Subgraph, Graph Mining, Verification
   - 로컬 h-클리크 밀도 기반 서브그래프 탐색은 중복 클리크와 검증 문제로 인해 수행이 어려운 문제가 있음.
   - 제안된 IPPV 파이프라인은 후보 생성, 분해, 검증 과정을 반복해 정확하고 빠르게 상위 k개의 LhCDS를 탐색함.

 - SPID-Join: A Skew-resistant Processing-in-DIMM Join Algorithm Exploiting the Bank- and Rank-level Parallelisms of DIMMs
   - Join Processing, Skew Resistance, Processing-in-Memory (PIM), DIMM Parallelism, UPMEM
   - 기존 PID 조인 알고리즘은 입력 테이블의 스큐로 인해 IDP 간 부하 불균형 문제가 있음.
   - SPID-Join은 DIMM의 뱅크 및 랭크 병렬성과 키 중복을 활용하여 스큐 문제를 해결함.

 - Directional Queries: Making Top-k Queries More Effective in Discovering Relevant Results
   - Top-k Queries, Skyline, Balanced Results, Directional Scoring, User Preference
   - 기존 Top-k 질의는 사용자 선호에 부합하는 스카이라인 튜플을 잘 반환하지 못하는 문제가 있음.
   - 튜플이 선호 방향에 얼마나 가까운지를 고려하는 방향성 질의를 제안하여 해당 문제를 해결함.

 - Connectivity-Oriented Property Graph Partitioning for Distributed Graph Pattern Query Processing
   - Graph Pattern Queries, Property Graphs, Partitioning, Crossing Matches, Distributed Systems
   - 기존 그래프 파티셔닝은 교차 매칭이 많아 분산 질의 시 성능 저하 문제가 있음.
   - 관계 라벨 기반의 약 연결 컴포넌트를 단위로 하는 RCP 기법으로 이 문제를 해결함.
 
 - On the Feasibility and Benefits of Extensive Evaluation
   - Benchmarking, Evaluation Sampling, ANOVA, Parameter Selection, Performance Prediction
   - 기존 성능 평가 설정 선택은 주관적이며, 모든 조합 실험은 비용이 너무 높은 문제가 있음.
   - 무작위 샘플링과 ANOVA 기반 예측 모델을 활용한 점진적 평가 방식으로 이 문제를 해결함.

 - BT-Tree: A Reinforcement Learning Based Index for Big Trajectory Data
   - Trajectory Indexing, Range Query, KNN Query, Cost Function, Reinforcement Learning
   - 기존 인덱스는 이동 경로 데이터에서 효율적인 범위 및 KNN 질의 처리를 잘 지원하지 못하는 문제가 있음.
   - 비용 함수 기반 이분 인덱싱과 강화학습 기반 전역 최적화를 결합한 BT-Tree로 이 문제를 해결함.

 - GABoost: Graph Alignment Boosting via Local Optimum Escape
   - Graph Alignment, Heterogeneous Graph, Local Optima, Boosting, Optimization
   - 기존 그래프 정렬 기법들은 복잡성으로 인해 국소 최적해에 머무르는 문제가 있음.
   - 초기 정렬 결과를 반복적으로 개선하여 더 나은 정렬 정확도를 제공하는 GABoost로 해결함.

 - Efficient and Accurate PageRank Approximation on Large Graphs
   - PageRank, Approximation Algorithm, Large Graphs, Matrix Decomposition, Scalability
   - 기존 PageRank 근사 기법은 계산 비용이 크거나 정확도가 낮고, 일부 정점에 대해서는 값이 제공되지 않는 문제가 있음.
   - CUR-Trans와 T²-Approx 알고리즘으로 정확도와 계산 효율을 높이면서 모든 정점에 대해 PageRank 값을 제공하도록 해결함.

 - Near-Duplicate Sequence Alignment with One Permutation Hashing
   - Near-Duplicate Detection, Sequence Alignment, Jaccard Similarity, One Permutation Hashing, Compact Windows
   - 기존 유사 중복 시퀀스 탐색 기법은 시퀀스 수가 많고 MinHash 스케치의 공간 비용이 커서 효율적인 처리가 어려운 문제가 있음.
   - One Permutation Hashing과 OPH Compact Window를 활용해 공간 비용을 O(n + k)로 줄이며 유사 시퀀스를 효율적으로 탐색하도록 해결함.

 - Efficient Approximation Algorithms for Minimum Cost Seed Selection with Probabilistic Coverage Guarantee
   - Seed Selection, Influence Maximization, Probabilistic Coverage, Approximation Algorithm, Social Networks
   - 기존 시드 선택 방식은 기대값 기반 도달 보장으로 인해 실제 캠페인 도달률이 목표에서 벗어나는 문제가 있음.
   - 확률 기반 도달 보장을 만족하면서 비용을 최소화하는 알고리즘을 설계하여 이 문제를 해결함.

 - Atom: An Efficient Query Serving System for Embedding-based Knowledge Graph Reasoning with Operator-level Batching
   - Knowledge Graph Reasoning, Embedding, Query Serving, Operator-level Batching, GPU Optimization
   - 기존 EKGR 시스템은 이질적인 질의로 인해 배칭 기회가 적어 온라인 질의 처리에서 비효율적인 문제가 있음.
   - 서로 다른 질의의 공통 연산자를 묶는 연산자 수준 배칭을 통해 질의 처리 효율성을 높이는 Atom 시스템으로 이 문제를 해결함.

 - A Profit-Maximizing Data Marketplace with Differentially Private Federated Learning under Price Competition
   - Federated Learning, Differential Privacy, Data Marketplace, Game Theory, Price Competition
   - 기존 데이터 마켓은 데이터 소유자가 가격을 설정할 수 없어 비현실적인 문제가 있음.
   - 가격 설정이 가능한 소유자를 포함한 3단계 Stackelberg 게임 기반 마켓플레이스로 이 문제를 해결함

 - Pluto: Sample Selection for Robust Anomaly Detection on Polluted Log Data
   - Log Anomaly Detection, Sample Selection, Transformer, Embedding Space, Robust Learning
   - 기존 로그 이상 탐지 모델은 이상 없는 로그가 필요하여 수작업 라벨링 비용이 큰 문제가 있음.
   - 오염된 로그에서 정제된 샘플을 자동 선택하여 강건한 이상 탐지를 가능하게 하는 Pluto를 제안함.

 - Adaptive Quotient Filters
   - Adaptive Filters, Quotient Filter, False Positive, Cache Efficiency, Auxiliary Structure
   - 기존 적응형 필터는 적응 보장이 약하고 오버헤드가 커서 실제 시스템에 사용되기 어려운 문제가 있음.
   - 적응 오버헤드를 최소화하고 강한 적응 보장을 제공하는 실용적인 적응형 필터 AdaptiveQF를 제안함.
   
 - A Lovász-Simonovits Theorem for Hypergraphs with Application to Local Clustering
   - Hypergraph Diffusion, Lovász-Simonovits Theorem, Conductance, Personalized PageRank, Local Clustering
   - 기존 하이퍼그래프 확산은 느리거나 비선형이라서 클러스터링에 적용하기 어려운 문제가 있음.
   - 평균 기반 확산과 APPRH를 활용해 이론적 컨덕턴스를 개선하고, A-HyperCut 알고리즘으로 효율적인 클러스터링을 해결함.

 - Optimizing LSM-trees via Active Learning
   - LSM-tree, Active Learning, Parameter Tuning, Key-Value Store, RocksDB
   - 기존 LSM-tree 튜닝은 수동적이며 변화하는 워크로드에 적응하기 어려운 문제가 있음.
   - 액티브 러닝 기반 Camal이 자동화된 튜닝과 동적 적응을 통해 성능 향상을 해결함.

 - Theoretically and Practically Efficient Maximum Defective Clique Search
   - Defective Clique, Graph Mining, Branching Algorithm, Pruning, Heuristic
   - 최대 k-결함 클리크를 찾는 문제는 NP-난해하며 기존 알고리즘은 비효율적인 문제가 있음.
   - 피벗 기반 분기 및 상한 가지치기 기법으로 시간복잡도를 개선하여 문제를 해결함.

 - Tao: Improving Resource Utilization while Guaranteeing SLO in Multi-tenant Relational Database-as-a-Service
   - Multi-tenancy, Database-as-a-Service, SLO Scheduling, Coroutine, Tasklet
   - 멀티 테넌트 DB 서비스에서 SLO를 보장하면서도 자원 활용률을 높이기 어려운 문제가 있음.
   - 태스크렛 기반 실행 구조와 SLO 스케줄러를 통해 두 목표를 동시에 달성하는 시스템으로 해결함.

 - Discovering Top-k Relevant and Diversified Rules
   - Rule Mining, Entity Enhancing Rules, Diversity, Relevance Learning, Approximation
   - 기존 규칙 탐색은 관련 없는 규칙이 과다하게 반환되는 문제가 있음.
   - 관련성과 다양성을 고려한 규칙 추출 알고리즘으로 이 문제를 해결함.

 - Enabling Adaptive Sampling for Intra-Window Join: Simultaneously Optimizing Quantity and Quality
   - Stream Join, Adaptive Sampling, Output Size, Variance, Performance Trade-off
   - 기존 샘플링 기반 조인은 출력 크기 감소 및 입력 의존성으로 인해 적응형 조정이 어려운 문제가 있음.
   - 실시간 스트림 조인을 위해 수량과 품질을 동시에 제어하는 적응형 샘플링 기법 FreeSam으로 이 문제를 해결함.

 - SketchQL: Video Moment Querying with a Visual Query Interface
   - Video Moment Retrieval, Visual Query Interface, Object Trajectories, Similarity Search
   - 자연어 또는 SQL 기반 질의는 사용 편의성과 범용성 측면에서 한계가 있는 문제가 있음.
   - 드래그 앤 드롭 기반 시각 질의와 객체 추적 기반 유사도 검색을 활용한 SketchQL로 이 문제를 해결함.
