+++
date = '2025-06-17T09:50:00+09:00'
title = 'Learned Index'
subtitle =  'Learned Index에 대한 직관을 얻기 위한 posting'
weight = 5
tags = ["RL", "Database"]
categories = ["Learned Index"]
+++

## **Learned Index 관련 paper 리서치**
### **The Case for Learend Index Structures (2018)**
- 1346회 인용 SIGMOD. google, MIT
- https://arxiv.org/pdf/1712.01208
- B-tree 등 Index 구조는 Model(Neural Network 포함하여) 변경될 수 있다는 개념을 구체화(증명?)
- RMI(Recursive Model Index) : Learned Index 소개 

추가적으로 공부해야하는 것들
- bloom filter
- CDF model
- B-tree (recap)
- Hash index (recap)
- 해당 논문의 RMI 이 대한 deep dive

### **Benchmarking Learned Indexes (2020)**
 - 185회 인용 VLDB. MIT, Intel, TUM(뮌휀 공과대학) 공동 연구
 - https://dl.acm.org/doi/10.14778/3421424.3421425
 - RMI(Recursive Model Index), RS(Radix Spline), PGM (Picevewise Geometric Model) 등과 같은 솔루션과 전통 솔루션들 벤치마크 제공
   - Learned Index 는 CDF를 근사한다  (참? 거짓?)
 - _읽기_ workload 에만 초점이 맞춰져 있어, _쓰기_ 워크로드는 future work으로 언급
 - PGM연구팀이 RMI보다 좋다는 언급에 긁혀서,, 쓴 논문으로 보임
 - Learned Index 를 평가할수 있는 benchmark에 대한 것, 각각의 size등 종합 평가.


### **The PGM-index: a fully-dynamic compressed learned index with provable worst-case bounds (2020) **
- 187회 인용 VLDB. Pisa (이탈리아) 연구
- https://dl.acm.org/doi/10.14778/3389133.3389135
- top-down 방식
- **아직 읽지 않음**


### **RadixSpline: A Single-Pass Learned Index (2020)**
- 114회 인용 SIGMOD. MIT, Intel, TUM(뮌휀 공과대학) 공동 연구
- 5 page
- https://dl.acm.org/doi/10.1145/3401071.3401659
- **아직 읽지 않음**


### **SOSD: A Benchmark for Learned Indexes (2019)**
- 118회 인용 VLDB. MIT, Intel, TUM(뮌휀 공과대학) 공동 연구
- https://arxiv.org/abs/1911.13014
- **아직 읽지 않음**


### **Learning Multi-dimensional Indexes(2019)**
 - https://mlforsystems.org/assets/papers/neurips2019/learning_nathan_2019.pdf

### **The Case for Learned Spatial Indexes**
 - https://arxiv.org/abs/2008.10349
 - 공간

### **ALEX: An Updatable Adaptive Learned Index**
 - https://arxiv.org/pdf/1905.08898

### **LSI: A Learned Secondary Index Structure**
 - https://dl.acm.org/doi/pdf/10.1145/3533702.3534912
  


### **A New Paradigm in Tuning Learned Indexes: A Reinforcement Learning-Enhanced Approach**
 -  SIGMOD 2025
 -  읽어봐야할거같은데
 -  Taiyi Wang (University of Cambridge)*; Liang Liang (Imperial College London); Guang Yang (Neo4j); Thomas Heinis (Imperial College); Eiko Yoneki (University of Cambridge)

### **VEGA: An Active-tuning Learned Index with Group-Wise Learning Granularity**
 - SIGMOD 2025
 -  읽어봐야할거같은데
 -  위에 것이랑 비슷하게 online tunning 을 가능하게 하는것 같은데..? 맞나
 

### **BT-Tree: A Reinforcement Learning Based Index for Big Trajectory Data**
 - SIGMOD 2025
 - 강화학습으로 이해


### **LITS: An Optimized Learned Index for Strings**
  - VLDB 2024
  - https://dl.acm.org/doi/10.14778/3681954.3682010

### **Morphtree: a polymorphic main-memory learned index for dynamic workloads**
  - VLDB 2025
  - https://dl.acm.org/doi/10.1007/s00778-023-00823-y


### **How good are multi-dimensional learned indexes? An experimental survey**
  - VLDB 2025
  -  https://dl.acm.org/doi/10.1007/s00778-024-00893-6