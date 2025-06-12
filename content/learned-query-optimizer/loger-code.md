+++
date = '2025-06-12T11:50:33+09:00'
draft = true
title = 'temp-LOGER 코드 분석'
subtitle =  'LOGER 프로젝트의 코드 분석 내용을 담은 내용입니다.'
weight = 4
tags = ["DBMS", "Database", "Optimizer", "Learned Query Optimizer", "Reinforcement Learning"]
categories = ["Learned Query Optimizer"]
+++



# **LOGER 프로젝트의 코드 분석 내용**

## **기본정보**
- LOGER: A Learned Optimizer towards Generating Efficient and Robust Query Execution Plans
- github repository
https://github.com/TianyiChen0316/LOGER


### **LOGER 실행을 위한 셋업**
```sh
# 3.8 이여야한다.
conda create -n loger python=3.8 -y
conda activate loger
pip install -r requirements.txt
pip install pandas
pip install dgl
pip install packaging
conda install libffi
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
pip install dgl-cu110==0.6.1

mkdir results
```

postgre 설치

{{% details title="아래내용은 balsa 프로젝트에 있는 내용" open=false %}}
```sh
cd ~
wget https://ftp.postgresql.org/pub/source/v12.5/postgresql-12.5.tar.gz
tar xzvf postgresql-12.5.tar.gz
cd postgresql-12.5

./configure --prefix=$HOME/postgresql-12.5 --without-readline
make -j
make install
echo 'export PATH=/home/jae.sim/postgresql-12.5/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
cd ~/
git clone https://github.com/ossc-db/pg_hint_plan.git -b REL12_1_3_7

cd pg_hint_plan/
vim Makefile
###
# Modify Makefile: change line
#   PG_CONFIG = pg_config
# to
#   PG_CONFIG = /home/jae.sim/postgresql-12.5/bin/pg_config

pg_ctl -D ~/imdb initdb
pg_ctl -D ~/imdb start -l logfile
```
{{% /details %}}

아래 내용을 추가
```python
# train.py
# 아래 내용을 추가
    parser.add_argument('--host', type=str, default='/tmp',
                        help='PostgreSQL host path')
```

### **LOGER 실행 커맨드**

```sh
# start postgre
pg_ctl -D ~/imdb start -l logfile

conda activate loger
## database  설정은 아래 내용 참조
 python train.py --database imdbload --port 5437 --host /tmp -U ""
```

```python
# train.py
    parser.add_argument('-D', '--database', type=str, default='imdb',
                        help='PostgreSQL database.')
    parser.add_argument('-U', '--user', type=str, default='postgres',
                        help='PostgreSQL user.')
    parser.add_argument('-P', '--password', type=str, default=None,
                        help='PostgreSQL user password.')
    parser.add_argument('--port', type=int, default=None,
                        help='PostgreSQL port.')
```

## **high level flow**

**train.py  .main**
 - 기초 Setup = train_set, test_set ready, database connect, log, cache 등을 설정,
 - 자체 클래스 `DeepQNet` 모델 initiailize
   -  `DeepQNet` 에는 Step1, Step2, PredictTail 총 세개의 Nueral Network를 가짐
   -  `Step1` : table level feature encoding
   -  `Step2` : 두 테이블간 embedding을 LSTM 기반으로 join representation을 생성
   -  `PredictTail` :  생성된 쿼리가 좋은지 판단하는 구조.
   -  요약
    > Step1: 테이블 임베딩 생성 (GNN) <br>
    > Step2: pairwise join composition (LSTM) <br>
    > PredictTail: partial plan value 예측 (value head) <br>
    > UseGeneratedPredict: 생성된 plan 검증 (classifier head)  <-- 안씀

{{% hint warning %}}
테이블 임베딩을 만들었으니, 새로운 워크로드 에 대해서는 테이블 임베딩이 많이 틀릴테니,
아예 예측 자체를 못하나?
{{% /hint %}}
 
**train.py  .train()**
 - Step1, Step2, PredictTail 개 를 training


// 미완

## **train_mode, test_mode of model**
DeepQNet은 모드별로 동작을 수행하도록 작성됨
```python
# LOGER/model/dqn.py
class DeepQNet:
    ...
    def train_mode(self):
        self.model_step1.train()
        self.model_step2.train()
        self.model_tail.train()

    def eval_mode(self):
        self.model_step1.eval()
        self.model_step2.eval()
        self.model_tail.eval()
```

## **workload sql 파싱**
1) LOGER는 sql을 파싱하는 로직을 별도의 .so 파일로 만들어두고 이를 import해서 사용함
```python
#  LOGER/core/sql.py
from psqlparse import parse_dict
...
class Sql:
  ...
            parse_result_all = parse_dict(self.sql)
  ...

```
여기에서 parse_dict는 아래와 같이 되어있으며, `parser.cpython-38-x86_64-linux-gnu.so` 는 프로젝트에 같이 탑재되어 있다.

```python
# LOGER/psqlparse/parser.py
def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import sys, pkg_resources, importlib.util
    __file__ = pkg_resources.resource_filename(__name__, 'parser.cpython-38-x86_64-linux-gnu.so')
    __loader__ = None; del __bootstrap__, __loader__
    spec = importlib.util.spec_from_file_location(__name__,__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
__bootstrap__()

```


## **상용 db 연결의 흔적**
oracle database를 사용한것으로 추정. 어느정도 구현이 되어있는지는 아직 분석 미완료
```python
# LOGER/train.py
...
    if args.oracle is not None:
        USE_ORACLE = True
        oracle_database.setup(args.oracle, dbname=args.database, cache=False)
    try:
        database.setup(dbname=args.database, cache=False)
...
```