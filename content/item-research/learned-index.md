+++
date = '2025-06-30T09:50:00+09:00'
title = 'Learned Index'
subtitle =  'Learned Index에 대한 직관을 얻기 위한 posting'
weight = 5
tags = ["RL", "ML", "NN", "Database"]
categories = ["Learned Index"]
+++

# **Learned Index 관련 paper 리서치**

## **Evaluation or Insight**
### **Learned Index: A Comprehensive Experimental Evaluation(2023)**
#### **Summary**
 - 62회 인용 VLDB. 칭화대
 - 그동안 나온 Learned Index를 망라하고, 다양한 벤치마크에서 성능을 비교 분석. <br>
 많은 항목에 대해서, Traditional Index가 더 뛰어나다는 것을 적시하고, Learned Index 의 구현별 장-단점을 총망라, 및 여러가지 Insight 제공 <br>
 - Learned Index의 목적 workload 구분
<img src="/images/lis_category_as_purpose.png" alt="lis_category_as_purpose" style="width:80%;" />
 - 아직 특정 워크로드들은 Traditional Index 방식이 더 성능이 뛰어남을 명시
<img src="/images/lis_trad_is_good.png" alt="lis_trad_is_good" style="width:80%;" />

#### ***주요 Insight***
  1) Learned Index는 <U>**복잡한 분포 or write-heavy workload**에 대해서는 Learned Index 의 advantage가 없음.</U>
  2) Learned Index는 <U>**range query 에 대해서도 advantage가 없음.**</U> (leaf node의 sorted data를 탐색하는데 시간을 많이 허비)
  3) Learned Index는 <U>**string key 로된 index 찾는 workload 에 대해서는 advantage가 없음**</U>
     - 후에 관련 연구가 나오긴 했음.(**LITS: An Optimized Learned Index for Strings(2024)** )
  4) Learned Index는 <U>**bulk loading 에 대해서 advantage가 없음**</U>
  5) linear model vs non-linear model에 대해<br>
  non-linear model이 대체로 더 높은 정확도를 제공하지만, training overhead가 있고, write operator에 slow-down 되는 특징이 있다.
  1) Learned Index는 큰 index 사이즈를 가지며, 추가적으로 buffer를 가지는데 (concurrency를 위해서), 이는 slow-down의 요인
  2) Learned Index는 <U>**concurrent lookup/insert workload에 대해서 advantage가 없음**</U><br><br>


#### **Introduction**
 - Learned Index 구조를 고려할때 필요한 세부적인 5개의 key factor 제시
   - Key Loopup
     - `position prediction`, `position search` 두 가지로 나뉨. (이 두개의 구현 algo 의 차이로 여러 variation이 있음) <br>
     `position prediction` 이 성공하면, key를 바로 리턴하지만, 틀릴경우 `position search`를 수행함
     - `position prediction` 관점 : hierarchy model of multiple 구조를 가짐. 1) key를 가진 자식 model을 예측하고, 2) 자식이 leaf node를 예측함.
     - `position search` 관점 : 선형 탐색(linear search - 오차가 작거나 법위가 작을때 유리)나, 이진탐색 (binary search - 탐색 범위가 넓을때 유리) 를 선택함
   - Key Insert
     - **_Key Insert 나 Key Delete를 지원하는 Learned Index 를 Mutable Learned Indexes 라고 지칭_** (지원하지 않는 Learned Index 도 많다..)
     - Insert 시에 모델의 재학습(retraining) 이 요구됨.
     - 보통 2가지 방식으로 해결, `In-place insert` 방식 or `Delta-buffer insert` 방식
     - `In-place insert` 는 index 사이에 gap을 reserve 해두어서, 모델의 재트레이닝을 지연시킴
     - `Delta-buffer insert`는 임시 버퍼에 저장하고, 나중에 batch insert해서 삽입 비용(구조 변경 및 재학습) 을 줄인다
   - Key Delete
     - Key Insert task와 비슷함
   - Concurrency 
     - intra-node concurrency 지원을 위해서 키마다 개별 buffer를 두고
     - across different node concurrency 지원을 위해서 temporary buffer 를 node마다 두어서<br> 1) merge/split 에서도 지원하고, 2) buffer를 merge 후에 합침
   - Bulk Loading
     - Top-down 방식과 Bottom-up 방식이 있고,
       - Top-down 방식은 root를 initialize하고, 각 child로 pair를 split하는것을 재귀적으로 하고,
       - Bottom-up 방식은 pair를 split한다음 leaf 노드에 할당하고, 각 노드의 min/max를 추출하고 이를 재귀적으로 처리하여 index를 구축한다..
     - split algorithm은 overhead를 고려한 여러 방식이 있다.

#### **Lookup Design**
 - Lookup desing 에 대해 한번에 찾으면 (prdiction 성공일 경우 O(1)의 시간복잡도, 실패시 position search 수행)
   - linear model은 크게 linear interpolation model 과 linear regression model (RMI 구조)로 나뉨. 
   - 추가 리서치 필요 --> (linear interpolation model 과 linear regression model)
   - non-linear model 은 polynomial fitting model과 neural network model(RMI구조)로 나뉨.
   - hybrid 모델로, RMI 구조의 first layer를 이용한뒤, second layer는 piecewise linear model을 이용한 것도 있음 = XIndex
 - non-linear model 이 대체로 좋은 성능(정확도)를 보임. 그러나 training 과 prediction 이 더 많은 시간이 소요. non-linear model이 더 많은 key를 노드에 가지고 있어 tree depth도 적다
 - position serach method 에 대해서, 
   - 항상 적확하게 position prediction 이 가능하게 만들던가 (LIPP)
   - prediction 하고, 틀리면 leaf-node를 re-search 하게 만들던가 (RMI)
   - prediction 하고, 틀리면 leaf-node와 internal node 둘다 re-search 하게 만든다.
 - position search 가 틀리면, 현재 예측했던 p' 기반으로, 이것보다 작으면 D[:p'] 를 탐색하고, 크면 D[p':N] 을 탐색한다. 
   - 이때 사용할 알고리즘이 여러개 있는데,
   - 선형탐색 (Linear search)는 오차가 작을수록 유리
   - 이진 탐색(binary search)는 오차가 클수록 유리한데, 변형으로 Exponential search와 Interpolation search가 있는데 데이터 분포에 따라 속도 차이가 있다.
 - [k_left, k_right] 를 찾는 range search에 대해서는, k_left 보다 큰 key를 찾아서, 소팅하여 k_right 보다 작은 key들을 정렬해서 찾는것 (= 사실상 key lookup의 조합으로 이해)
 - duplicated key 의 경우, 예측(prediction)이 정확하게 맞더라도 추가적인 탐색을 수행한다.

#### **Insert Design**
 - leaf node 가 가질 pair 가 threshhold 를 넘거나, data fitting 이 low quality이라면 structural modification을 수행한다.
   - Insert는 두가지 전략이 있는데, `delta-buffer insert` 전략과 `in-place insert` 전략이 있다.
   - `delta-buffer insert`는 index-level, node-level, pair-level buffer를 각기 가지도록 구현될 수 있는데,<br> pair-level은 더 높은 concurrency performence를 주지만, extra storage overhead를 야기한다.
   - `in-place insert`는 node 사이의 gap을 reserve 하는 전략인데. gap 공간이 비어있으면 바로 insert, 아닌 경우에는 conflict을 해결하는 두 전략이 있다.
     - target position 과 closted gap에 pair를 shift 하기
     - 새로운 new node에 inserted pair 와 exsiting pair를 넣고, new node를 target postion에서 가르키기
 - structural modification 방법에 대해서 4가지 방법
   - Fullness-based method (포화도 기반) : node나 버퍼의 임계값 초과시 structural modification 수행
   - Error-based method (오차 기반) : 모델의 예측 오차가 임계값을 초과 할 경우 structural modification 수행
   - Cost-based method (비용 기반) : ALEX의 경우 3가지 전략 소개. 이중에 가장 cost가 덜드는 것을 선택 (average latency of lookup/insert for each leaf node)
     - Expand the node : node가 더 많은 key pair를 담을 수 있도록 공간을 확장하고, 키를 재배치
     - Split the node : leaf 노드를 두개의 node로 나뉘고 각각 학습. new leaf node는 같은 parent node에 연결
     - Rebuild the node : bulk loading algorithm 을 이용해서 sub-tree를 build하고 통째로 교체
   - Conflict-based method (충돌 기반) : 충돌 쌍 개수가 threshold 초과 할 경우 structural modification 수행

#### **Delete Design**
 - Insert design과 유사. 언제 structural modification 이 수행해야할지 조건을 설정하고, node를 merge함.

#### **Concurrency Design**
 - intra-node concurrency 와 inter-node concurrency가 있다.
   - intra-node concurrency는 키마다 buffer를 두어서 해결하고, (node-level, index-level, pair-level에 둘수 있음)
   - inter-node concurrency는 Temporary-buffer를 두거나, Buffer-Train-merge 전략을 사용.
     - Temporary-buffer는 분할이 진행되는 동안에, temporary buffer로 delta 를 들고 있다가, split 작업이 완료되면 새 node에 insert하도록 구현
     - Buffer-train-merge 는 버퍼가 가득차면, train 하고, merge 한다는 전략으로, pair 쌍 buffer에 대해서 할수 있고, sub-node level에서도 할 수 있다.

#### **Bulk loading Design**
 - Top-down bulk loading 과 Bottom-up bulk loading 두 가지 방식이 있다.
 - Top-down bulk loading은 root 노드에 다 넣어보고, split 해야하면, 재귀적으로 점차 쪼개가는 방식. 두가지 challenge가 있다
   - split 조건선정
   - 얼마나 분할할지(얼마나 많은 child node), 어떻게 pair를 할당할지
 - Bottom-up bulk loading은 전부 leaf nod로 쪼개고, minimal과 maximal key를 입력으로 상위 노드를 만들어감. Top-down 과 동일한 challenge 공유
 - split 조건 선정으로는 `Greedy Split`, `Even Split`, `Cost-based Split`, `Confilt-based Split`이 있다.
 - Top-down vs Bottom-up
   - Bottom-up은 perdiction error 가 발생 가능, Top-down은 발생 하지 않음
   - Top-down은 보통 monotonic model로 구현되고, non-monotonic이면 다음 두가지 문제가 있다고한다. (모델예측 오차가 크고, 위치탐색 비용이 증가)
 - Other Method로는, RMI는 non-tree structure 이다. (two-layer structure를 사용)


### **The Case for Learend Index Structures (2018) : RMI 소개 및 Learned Index 개요 소개**

#### **Summary**
- 1346회 인용 SIGMOD. google, MIT
- https://arxiv.org/pdf/1712.01208
- B-tree 등 Index 구조는 Model 변경될 수 있다는 개념을 구체화(증명)
- LIF(The Learning Index Framework), RMI(Recursive Model Index), standard-error-based search strategy 소개

추가적으로 공부해야하는 것들
- bloom filter
- CDF model
- B-tree (recap)
- Hash index (recap)
- 해당 논문의 RMI 이 대한 deep dive

#### **Detail**
 - B-Tree는 ML의 position prediction으로(regression tree), Bloom filter는 binary classify로 대체될 수 있다는 직관을 제시
   - B-Tree 자체가 이미 오차를 포함하고 있어서(최대-최소 오차만 guarantees함) regression tree 로 대체가 가능하고, Neural Network 나 다른 ML 모델로도 대체가 가능
   - ML 모델로 행한 prediction이 다소 부정확하더라도, local search 하면 쉽게 보정할 수 있다.
 - GPU에 탑재된다면 가속기를 이용해서 더 빨라질 수 있는 점을 언급
 - ***Indexing은 본질적으로 데이터 분포를(CDF)를 학습하는 문제*** 라는 insight를 제시
 - LIF에 대해서, 학습을 용이하도록 하는 framework를 개발한것으로 파악함.(model 이식, extract, configuration export 등을 지원). 중요 내용이 아니므로 이하 생략 
 - **MoE(Mixture of Experts)의 영감을 받아 recursive regression model 을 제시**. 이는 다음과 같은 장점을 지님<br>
  <img src="/images/lis_recursive_regression_model.png" alt="lis_trad_is_good" style="width:80%;" />
   1) model size and complexity 와 execution cost 의 분리
   2) 전체 학습보다 쉽다. easy to learn the overall shape of the data distribution
   3) divides the space into sub-range.( 적은 operator로 last mile searach 정확도를 확보)
   4) stage 간에 search process 가 필요 없다
 - recursive model의 장점은 다양한 ML을 선택 할 수 있다는 점이고. 하위 layer에는 B-Tree 마져 선택 할 수 있다.
   - 이는 하한이 보장되는 모델 설계가 가능하다

 - Range Search는 하한과 상한을 구해야하는 문제로, ML을 적용한 Index일 경우 모델이 monotonic(단조) 이여야 한다 (보통 monotonic 하게 구현)
   - 아닐경우, 잘못된 upper/lower bound를 return
   - ML의 monotonic 을 강제하는 기존 연구를 활용
     - M. Gupta, A. Cotter, J. Pfeifer, K. Voevodski, K. Canini, A. Mangylov, W. Moczydlowski, and A. Van Esbroeck. Monotonic calibrated interpolated look-up tables. The Journal of Machine Learning Research, 17(1):3790–3836, 2016.
     - S. You, D. Ding, K. Canini, J. Pfeifer, and M. Gupta. Deep lattice networks and partial monotonic functions. In NIPS, pages 2985–2993, 2017

 - 문자열 key에 대해서는 tokenization을 진행(입력 feature로 변경)해서 index 학습
   - 옛날 논문이므로 string 에 대해서 미래 연구 여지가 있다고 언급
 - 트레이닝은 매우 빠르다.(수 초내 수요)
 - ***Hash Map Index를 ML 로 대체하는 것은 추가 연구가 필요하다고 언급***
 - Bloom filter는 no false negative(FNR =0) 그러나 false positive(FPR <= 일정수준)가 허용됨
   - binary probabilistic classification 문제로 해결. RNN, CNN 등을 적용 가능
   - 특정 임계값 이상일 경우 키가 존재한다고 판단
   - 그러나 이는 FPR, FNR이 0이 아님 overflow bloom filter를 도입(키가 존재한다면 여길 추가 확인)
 
### **Benchmarking Learned Indexes (2020)**
 - 185회 인용 VLDB. MIT, Intel, TUM(뮌휀 공과대학) 공동 연구
 - https://dl.acm.org/doi/10.14778/3421424.3421425
 - RMI(Recursive Model Index), RS(Radix Spline), PGM (Picevewise Geometric Model) 등과 같은 솔루션과 전통 솔루션들 벤치마크 제공
   - Learned Index 는 CDF를 근사한다 
 - _읽기_ workload 에만 초점이 맞춰져 있어, _쓰기_ 워크로드는 future work으로 언급
 - PGM연구팀이 RMI보다 좋다는 언급에 긁혀서,, 쓴 논문으로 보임
 - Learned Index 를 평가할수 있는 benchmark에 대한 것, 각각의 size등 종합 평가.
 - 후에 나온 **Learned Index: A Comprehensive Experimental Evaluation(2023)** 으로 대신하면 될것 같음
 

## **Single dimention**
### **The PGM-index: a fully-dynamic compressed learned index with provable worst-case bounds (2020)**
- 187회 인용 VLDB. Pisa (이탈리아) 연구
- https://dl.acm.org/doi/10.14778/3389133.3389135
- RMI 대비 top-down 방식.
- Learned Index의 초기방식으로, RMI비교 분석을 하고 있음.


### **RadixSpline: A Single-Pass Learned Index (2020)**
- 114회 인용 SIGMOD. MIT, Intel, TUM(뮌휀 공과대학) 공동 연구
- 5 page
- https://dl.acm.org/doi/10.1145/3401071.3401659
- **아직 읽지 않음**


### **SOSD: A Benchmark for Learned Indexes (2019)**
- 118회 인용 VLDB. MIT, Intel, TUM(뮌휀 공과대학) 공동 연구
- https://arxiv.org/abs/1911.13014
- **아직 읽지 않음**


### **A New Paradigm in Tuning Learned Indexes: A Reinforcement Learning-Enhanced Approach**
 -  SIGMOD 2025
 -  읽어봐야할거같은데
 -  Taiyi Wang (University of Cambridge)*; Liang Liang (Imperial College London); Guang Yang (Neo4j); Thomas Heinis (Imperial College); Eiko Yoneki (University of Cambridge)

## **Multi dimention**
### **Learning Multi-dimensional Indexes(2020)**
 - https://mlforsystems.org/assets/papers/neurips2019/learning_nathan_2019.pdf

### **Tsunami: A Learned Multi-Dimensional Index for Correlated Data and Skewed Workloads**
 - multi dimention

### **LSI: A Learned Secondary Index Structure**
 - https://dl.acm.org/doi/pdf/10.1145/3533702.3534912


### **VEGA: An Active-tuning Learned Index with Group-Wise Learning Granularity**
 - SIGMOD 2025
 -  읽어봐야할거같은데
 -  위에 것이랑 비슷하게 online tunning 을 가능하게 하는것 같은데..? 맞나
 

### **BT-Tree: A Reinforcement Learning Based Index for Big Trajectory Data**
 - SIGMOD 2025
 - 강화학습으로 이해


### **How good are multi-dimensional learned indexes? An experimental survey**
  - VLDB 2025
  -  https://dl.acm.org/doi/10.1007/s00778-024-00893-6

## **String Task**
### **LITS: An Optimized Learned Index for Strings**
  - VLDB 2024
  - https://dl.acm.org/doi/10.14778/3681954.3682010


## **Updatable**
### **ALEX: An Updatable Adaptive Learned Index(2020)**
 - https://arxiv.org/pdf/1905.08898
 - MIT Krask 교수팀


## **Spatial Index**
### **LISA:ALearnedIndex Structure for Spatial Data**
 - Spatial index. 공간

### **Effectively Learning Spatial Indices**
  - 공간

### **The Case for Learned Spatial Indexes**
 - https://arxiv.org/abs/2008.10349
 - 공간

## **Extra**
### **Morphtree: a polymorphic main-memory learned index for dynamic workloads**
  - VLDB 2025
  - https://dl.acm.org/doi/10.1007/s00778-023-00823-y