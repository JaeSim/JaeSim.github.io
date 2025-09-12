+++
date = '2025-08-27T14:05:14+09:00'
title = 'Milvus_build from code'
weight = 8
tags = ["Vector", "Database", "Embedding", "ANN", "Similarity Search"]
categories = ["Vector database","Study", "development"]
+++

# **Milvus를 build 하기**
## **기본 정보**
 - https://github.com/milvus-io/milvus
 - 아래 요구사항
 - go와 c++로 작성된 프로젝트로써, 사용시에는 보통 python SDK `pymilvus`를 이용해서 사용한다
 ```sh
 go: >= 1.21
 cmake: >= 3.26.4
 gcc: 9.5
 python: > 3.8 and  <= 3.11
 ```

### **준비물**
#### **conda activate 및 git clone**
 - conda 가상환경 설정 및 git clone
 ```sh
 conda create -n vec python=3.8
 conda activate vec

 git clone https://github.com/milvus-io/milvus
 ```
#### **go 설치**
 - https://go.dev/doc/install download Go file 또는 아래 명령어로 설치
 ```sh
 wget https://go.dev/dl/go1.25.0.linux-amd64.tar.gz
 ```
 - 기존 go project 지우고, 압축풀기
 ```sh
   # delete old go project
   rm -rf /usr/local/go && tar -C /usr/local -xzf go1.25.0.linux-amd64.tar.gz
 ```
- path 설정 및 확인
```sh
export PATH=$PATH:/usr/local/go/bin
vim ~/.bashrc
go version
```
### **Build**
 - 준비물
 ```sh
 python3 -m pip install "conan>=2,<3"
 export PATH="$HOME/.local/bin:$PATH"

 conda install -c conda-forge "cmake>=3.22" ninja
 . "$HOME/.cargo/env"
 ```
 - 빌드스크립트
 ```sh
 cd milvus
 ./scripts/install_deps.sh
 make SKIP_3RDPARTY=1
 ```

  - 중간에 에러가 나면
  ```sh
go get -u github.com/bytedance/sonic@v1.14.0
go mod tidy
  ```

### **Start**
 - start commend
```sh
./scripts/start_standalone.sh
```
 - /tmp/standalone.log 에 에러가 쌓임