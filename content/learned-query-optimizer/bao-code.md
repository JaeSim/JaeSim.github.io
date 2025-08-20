+++
date = '2025-07-04T13:48:30+09:00'
title = 'BAO 코드 분석'
subtitle =  'BAO 프로젝트의 코드 분석 내용을 담은 내용입니다.'
weight = 5
tags = ["DBMS", "Database", "Optimizer", "Learned Query Optimizer", "Reinforcement Learning", "BAO"]
categories = ["Learned Query Optimizer"]
+++

# **BAO 프로젝트의 코드 분석 내용**

## **기본정보**
- Bao: Making Learned Query Optimization Practical
- github repository
https://github.com/learnedsystems/BaoForPostgreSQL

### **BAO 실행을 위한 셋업**
 
- extension 설치
```sh
cd pg_extension
sudo chown -R $(whoami) /home/{User}/data/postgresql-12.5/
make USE_PGXS=1 install
```
- postgresql.conf 에 shared_preload_libraries 에 추가하기

  ```conf
  # postgresql.conf
  # shared_preload_libraries = 'pg_hint_plan'
  shared_preload_libraries = 'pg_bao'
  ```
 
    - bao는 하나의 hook만 가질수 있도록 되어있음
    ```c++
      // extension/main.c
      if (prev_planner_hook) {
        elog(WARNING, "Skipping Bao hook, another planner hook is installed.");
        return prev_planner_hook(parse, cursorOptions,
                                boundParams);
      }
    ```

- postgre 시작 (balsa github) 및 확인
```sh
pg_ctl -D ~/imdb start -l logfile
psql imdbload -p 5437
```
```sql
imdbload=# SHOW enable_bao;
 enable_bao
------------
 off
(1 row)
```

- conda 세팅
```sh
cd {bao path}
conda create -n bao python=3.8 -y
conda activate bao
pip3 install scikit-learn numpy joblib
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
# for mssql
pip install pyodbc
```

- bao_server 시작
```sh
cd bao_server
python3 main.py
```
- 확인 커맨드. 아래 커맨드 시작시 bao_server log에 `Logged reward of 2103.556027` format의 글귀가 나옴
```sh
SET enable_bao TO on;
SET
EXPLAIN SELECT count(*) FROM title;
SELECT count(*) FROM title;
```


## **내부 특성**
### **학습 범위**
 - ***BAO는 Join order에 대해서 학습하지는 않음***
 - 전체 쿼리에 대해서 Join Op, Scan Op를 강제할지 정함
 - 지원하는 hint list. ***default는 모두 ON***
 ```sql
 SET enable_nestloop TO off;
 SET enable_hashjoin TO off;
 SET enable_mergejoin TO off;
 SET enable_seqscan TO off;
 SET enable_indexscan TO off;
 SET enable_indexonlyscan TO off;
 ```

### **pg_extension으로 connector layer(가칭) 구현**
 - 일부 기능이 ***postgres pg_extension 으로 구현됨*** (=postgres에 강하게 커플링 되어있음). c언어
 - **기능1**: SQL을 호출하면 extension 이 hook으로 먼저 받아서, original SQL을 실행전에 hint statement를 주입하는 동작을 한다.
    - 예시
        - 기존 : 1) SQL 호출 --> 2) DBMS SQL 실행
        - BAO : 1) SQL 호출 --{extension hook}--> 2) extension에서 candidate plan 생성 및 buffer 정보 획득 후 bao_server에 전달 --> <br> 
          3) bao_server plan중 best 선택 후 response --> 4) extension 에서 선택된 plan에 맞는 hint statement 생성 --> 5) hint statement 수행 --> 6) DBMS SQL 실행
 - **기능2**: 이후 나온 결과를 BAO Server에도 전달한다 (latency 전달)

### **모델 및 cli 등은 python으로 구현**
 - python layer에서는 학습된 parameter 들이 있고, trainset load 및 SQL 호출등은 python 모듈의 layer에서 실행됨
   - 이때 전달되는 paramter는 boolen으로 전달되는것으로 보이고, connector layer 에서 boolen을 hint statement로 변환함.


### **json 형태의 message interface**
 - json을 보내기 전에 해당 message가 어떤 meesage type인지를 보내고, 마지막에 terminate_messgae를 보낸다
 - 결국 START_XXXX_MESSAGE, JSON_CONTENT, TERMIMNAL_MESSAGE 로 구성되어 있고, TERMIAL_MESSAGE 를 받을때까지 que에 저장한뒤 수신시 처리한다.
 ```python
START_QUERY_MESSAGE = "{\"type\": \"query\"}\n"
START_FEEDBACK_MESSAGE = "{\"type\": \"reward\"}\n"
START_PREDICTION_MESSAGE = "{\"type\": \"predict\"}\n"
TERMINAL_MESSAGE = "{\"final\": true}\n"
 ```

### **Shared Buffer 활용**
 - Bao 논문에는 따로 언급되어 있지 않지만, DBMS의 buffer에 있는 조건을 학습에 사용하고 있음
   - 위치 pg_extension/bao_bufferstate.h
   ```c
   static char* buffer_state() {
   ...
   ```
   - 위 함수를 통해 buffer륻 획득한후에 json에 적재한다.
   ```json
    {
      "Plan": {
        "Node Type": "Other",
        "Node Type ID": "42",
        "Total Cost": 50166.515833,
        "Plan Rows": 1,
        "Plans": [
          {
            "Node Type": "Other",
            "Node Type ID": "45",
            "Total Cost": 50166.500833,
            "Plan Rows": 2,
            "Plans": [
              {
                # skip... 
              }
            ]
          }
        ]
      },
      "Buffers": {
        "title_pkey": 1,
        "kind_id_title": 1
      }
    }
   ```


### **bandit arm 5 짜리 문제로 치환**
 - BAO 는 학습범위의 힌트들에 대해서 2^6 개의 모든 조합에 대해서 search space로 두지 않고, <br>
 ***기본적으로 5개의 조합에 대해서만 검사한다***
 ```c
#define BAO_MAX_ARMS 26

// Each Bao config variable is linked to a PostgreSQL session variable.
// See the string docs provided to the PG functions in main.c.
static bool enable_bao = false;
static bool enable_bao_rewards = false;
static bool enable_bao_selection = false;
static char* bao_host = NULL;
static int bao_port = 9381;
static int bao_num_arms = 5;
static bool bao_include_json_in_explain = false;
#endif
...
 static void set_arm_options(int arm) {
  enable_hashjoin = false;
  enable_mergejoin = false;
  enable_nestloop = false;
  enable_indexscan = false;
  enable_seqscan = false;
  enable_indexonlyscan = false;
  
  switch (arm) {
  case 0:
    enable_hashjoin = true;
    enable_indexscan = true;
    enable_mergejoin = true;
    enable_nestloop = true;
    enable_seqscan = true;
    enable_indexonlyscan = true;
    break;
    
  case 1: 
    enable_hashjoin = true; 
    enable_indexonlyscan = true; 
    enable_indexscan = true; 
    enable_mergejoin = true; 
    enable_seqscan = true; 
    break;
  case 2: 
    enable_hashjoin = true; 
    enable_indexonlyscan = true; 
    enable_nestloop = true; 
    enable_seqscan = true; 
    break;
  case 3: 
    enable_hashjoin = true; 
    enable_indexonlyscan = true; 
    enable_seqscan = true; 
    break;
  case 4: 
    enable_hashjoin = true; 
    enable_indexonlyscan = true; 
    enable_indexscan = true; 
    enable_nestloop = true; 
    enable_seqscan = true; 
    break;
  case 5: 
    enable_hashjoin = true; 
    enable_indexonlyscan = true; 
    enable_mergejoin = true; 
    enable_nestloop = true; 
    break;
    ...
 ```

