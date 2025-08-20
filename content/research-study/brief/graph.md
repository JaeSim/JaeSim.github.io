+++
date = '2025-06-13T14:45:35+09:00'
title = 'GNN'
subtitle =  'GNN에 대한 직관을 얻기 위한 posting'
weight = 4
tags = ["Neural Network", "GNN", "추천시스템"]
categories = ["Graph Neural Network","Study"]
+++



# **GNN 직관**


> CNN은 Euclidean 공간에서 행과 열로 배열된 픽셀들로 이루어진 이미지에서 특징을 추출한다. <br>
> LSTM은 자기 자신으로 돌아오는 recurrent 구조를 통해 input으로 주어지는 어떤 한 시계열 데이터인 sequence의 특징을 추출한다. <br>
> Transformer는 self-attention 구조를 통해 어떤 한 부분에서 주의를 기울여야 할 여러 부분을 병렬적으로 함께 처리함으로써 input의 특징을 추출한다. <br>
> 감이 오겠지만 마찬가지로 GNN에서도 그래프 구조를 통해 시스템에서의 관계 등 다양한 특징을 추출한다. 


## **keywords**

- Graph representation learning
   - 먼저 시작되고 여기에 NN이 접목된것이 GNN
- Unstructured data
   - GNN 은 Unstructured data에 주로 사용됨
- Node embedding (Shallow embedding, Message passing GNN)
   - 적은 차수의 데이터로 projection 하는것. 지금은 GNN은 주로 노드 기반의 임베딩이 함, edge 기반 임베딩은?
- Aggregation
   - embedding 된 정보를 잘 합치는것
- Neighborhood and adjacency matrix
   - Neighborhood : 정의를 어떻게 하느냐에 따라 성능이 크게 좌우됨
- (Neighborhood) Attention


## **참고**

https://glanceyes.com/entry/%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-GNNGraph-Neural-Network%EC%99%80-%EC%9D%B4%EB%A5%BC-%EC%9D%91%EC%9A%A9%ED%95%9C-NGCFNeural-Graph-Collaborative-Filtering%EC%99%80-LightGCN

https://velog.io/@tobigs-gnn1213/Limitations-of-Graph-Neural-Networks

https://www.youtube.com/watch?v=rUmRlZzD_Uk