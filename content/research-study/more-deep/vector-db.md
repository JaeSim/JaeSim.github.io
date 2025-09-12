+++
date = '2025-08-06T10:45:46+09:00'
title = 'Vector DB'
weight = 8
tags = ["Vector", "Database", "Embedding", "ANN", "Similarity Search"]
categories = ["Vector database","Study"]
+++

# **Vector Database 관련 학습 내용**
 - 본인의 이해/기억을 위해 작성한 것이므로, 많은 내용이 생략, 축약되어 있음
## **기본 정보**
 - Vector database는 vector의 유사도 검사를 빠르게 하기 위한것
 - 비정형 데이터의 semantically 응답 가능
 - 질의했을때 유사한 vector을 top-k 개 만큼 반환하는 것
 - 현재로써는 RAG에서 응용되는것이 주된 사용처
 - RAG flow
 <img src="/images/rag_flow.png" alt="lis_trad_is_good"/>

 [그림출처: Retrieval-Augmented Generation for Large Language Models: A Survey (https://arxiv.org/abs/2312.10997)] 
 - 보통 벡터 임베딩과 메타데이터로 구성되어 있음
 - 최근접 이웃 (KNN)은 매우 느려 정확하진 않지만 충분히 정확한 ANN을 사용
 - 유사도 측정시 cosine similarity, Euclidean Distance(L2), Dot Product(내적) 계산 방식이 있음
 - CromaDB(RAG에 최적화), Milvus 등이 있다
 - FAISS (Facebook AI Similarity Search) 는 라이브러리 이므로 참조할 것
 
## **한계**
 - 정확한 값 매칭에 비효율
 - 복잡한 관계형 쿼리 미지원
 - transaction 기능이 약함 ACID
 - embedding 모델의 성능이 검색 품질 좌우

## **Vector Search Solution**
### **Milvus: A Purpose-Built Vector Data Management System (SIGMOD 2021)** 
- vector database
- Faiss 대비 고차원 및 large scale에서 속도향상, 복잡한 쿼리 지원
- 복잡한 쿼리를 지원하며 (partition-based strategy), multi-vector 쿼리 처리에 대해 좋다고 언급 (vector fusion, iterative merge)
- Distributed environment (cloud 환경) 에서 적합한 구조
- Open-source C++ library
- LSM-Tree based 방식으로 삽입과 삭제를 지원
- Snapshot isolation으로 read-write 충돌 우회 <br>
  질의가 들어온 시점에서 현재 LSM-Tree에 있는 메모리와 디스크를 snap샷으로 고정하고 여기서 질의
- Graph-based index(HNSW, RNSG) 와 quantization-based index(IVF_FLAT, IVF_SQ8, IVF_PQ)를 지원
- CPU-GPU co-design과 메모리 구조 등을 아우르는 최적화

### **Faiss(Facebook AI Similarity Search)**
 - 고성능 벡터검색을 위한 ***라이브러리 (DB 아님)***
 - GPU 가속을 사용가능

### **pgvector**
 - postgresql 에서 벡터 임베딩 기반 검색을 지원하기 위한 extension.
 - 추가적으로 설치가 필요.
 - 대규모(수억 개 이상) 데이터 처리에는 Faiss, Milvus, Weaviate 같은 전용 벡터DB보다 성능이 제한

## **Vector Embedding**
 - 텍스트나, 이미지 등에서 semantic 특징을 추출하여 숫자 array [벡터] 로 표현
 - workd2vec, sentence-BERT, CLIP 등
 - vector db는 embedding vector를 효율적으로 query하는 database
 - 고차원 숫자 배열

## **Vector DB Index**
 - ANN 검색 속도 향상을 위한 특화 index
### **HNSW : Hierarchical Navigable Small World**
<img src="/images/vec_hnsw.png" alt="hnsw_image" style="width:50%;"/>
 - graph 기반
 - 순차적으로 node를 삽입해내가며, 만들고, 삽입시에 layer의 level을 랜덤하게 결정한다 (지수분포에 따라서). 하위로 갈수록 노드갯수가 많아지고, 최상위 상위는 몇개없음.

### **IVF : Inverted File Index**
 - clustering 기반
 - 여러개의 중심점(centroid)로 나눈뒤 각 cluster안에서만 찾아서 일부만 탐색하는 기법. IVF-PQ, IVF-HMSW등과 같이 사용

### **LSH : Locality-Sensitive Hashing**
 - Hashing 기반
 - 유사도가 높은 vector를 같은 해쉬 버킷에 들어가도록 하는 버킷을 생성.
 - 이것은 확률적으로 근사를 보장하고, 고차원데이터일 경우 해쉬 테이블 비용이 증가하는 단점이 있음(차원이 높으면 유사도가 낮아지니 테이블이 많아짐).

### **PQ : Product Quantization**
 - 벡터 압축 기반
 - vector를 작은 벡터(subvector)로 구역을 나누고, 전체 데이터로부터 효율적인 압축이 될수있는 코드북을 **학습** 한다. 따라서, 데이터분포변화에 대해서 취약
 - 이후 코드북으로 압축

## **참조**
 - Survey of vector database management systems, VLDB 2024
 - A Comprehensive Survey on Vector Database: Storage and Retrieval Technique, Challenge
 - Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs  (HNSW)