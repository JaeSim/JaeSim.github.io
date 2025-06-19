+++
date = '2025-06-12T11:50:33+09:00'
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
- 기본 패키지 설치
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
pip install pyodbc

mkdir results
```

- postgre 설치

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


# 실행확인
psql imdbload -p 5437
```
{{% /details %}}


- postgre 에 dataset ready ssb <br>
타 db로 테스트하여도 로직상 postgre에 데이터셋이 설정되어 있어야한다.
{{% details title="ssb setup 내용_postgre based on balsa" open=false %}}

https://github.com/nuko-yokohama/ssb-postgres/tree/master
```sh
# balsa 프로젝트에 있는 파일 을 이용
# Create and start the DB
pg_ctl -D ~/imdb initdb`

# Copy custom PostgreSQL configuration.
cp ~/balsa/conf/balsa-postgresql.conf ~/imdb/postgresql.conf

# Start the server
pg_ctl -D ~/imdb start -l logfile

# SSB DB 만들기
createdb -p 5437 ssb

# SSB용 테이블 정의
psql ssb -f tables.sql -p 5437

# 데이터셋 생성
cd ssb-dbgen
./dbgen -s <scale factor> -T a

# 생성된 데이터셋 파일 확인
`ls -1 *.tbl`

#customer.tbl
#date.tbl
#head-customer.tbl
#head-lineorder.tbl
#lineorder.tbl
#part.tbl
#supplier.tbl


# 데이터셋 로드
cd ..
psql ssb -f load.sql -p 5437
# load.sql에서 경로를 data 폴더 절대경로를 잘 지정해줘야한다.
```
{{% /details %}}


- 아래 내용을 추가
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
## postgres
python train.py --database imdbload --port 5437 --host /tmp -U ""
## mssql
python train.py --database ssb --port 5437 --host /tmp -U "" --mssql "mssql" --dataset dataset/ssb_train dataset/ssb_test
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

## **SQL parse and composition**

```python
# LOGER/core/sql.py
class Sql:
    _re_like = re.compile(r'^%([^%]+)%$')

    def __init__(self, sql, feature_length=2, filename=None, device=torch.device('cpu'), table_space=None):
        ...
        self.parse_join()
        ...
    def __str__(self):
        if isinstance(self.sql, str):
            return self.sql
        return self.__str()
        ...
    def __str(self):
        edges = list(map(lambda x: x[2], self.edge_list))
        ...
```

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

### **상용 db을 위한 postgre 세팅**


LOGER는 oracle로 테스트 한것으로 보이는데, 독특한점은 oracle로 train을 시작해도 postgre에 동일한 dataset이 ready되어 있어야한다. <br>
아마 postgre 에 먼저 구현한뒤에 급하게 oracle을 붙여서 그런것 같다..
```python
    if args.oracle is not None:
        # USE_ORACLE = True
        db_engine = 'oracle'
        oracle_database.setup(args.oracle, dbname=args.database, cache=False)
    elif args.mssql is not None:
        # USE_ORACLE = True
        db_engine = 'mssql'
        mssql_database.setup(args.mssql, dbname=args.database, cache=False)

    # TODO: database.setup and mssq-database.setup are difference layer.
    # Even we want to test mssql, postgres must be ready

    try:
        database.setup(dbname=args.database, cache=False)
    except:
        try:
            database_args = {'dbname': args.database}
            if args.user is not None:
                database_args['user'] = args.user
            if args.password is not None:
                database_args['password'] = args.password
            if args.port is not None:
                database_args['port'] = args.port
            if args.host is not None:
                database_args['host'] = args.host
            database.setup(**database_args, cache=False)
        except:
            database.assistant_setup(dbname=args.database, cache=False)

```



## **작업일지**

1) mssql.py 생성 (based oracle.py)
2) mssql.py 에 pyodbc 를 이용해서 동작하도록 일부 포팅
3) USE_ORACLE 있는 부분을 db_engine 으로 대체
4) 실행 커맨드 옵션 추가
```python
    parser.add_argument('--host', type=str, default='/tmp',
                        help='PostgreSQL host path')
    parser.add_argument('--mssql', type=str, default=None, # LOCALDSN in mssql.py
                        help='To use mssql with given connection settings.')
  ```
5) step1.py 에 아래 로직 추가
```python
# step1.py
        table_others = g.nodes['table'].data['others'].float()
        # table_others = g.nodes['table'].data['others']
#sql.py
 # g.nodes['table'].data['others'] = x_dict['table_others']
 g.nodes['table'].data['others'] = x_dict['table_others'].to(torch.float32)
```
{{% details title="에러내용" open=false %}}
```sh
Traceback (most recent call last):
  File "train.py", line 830, in <module>
    train(beam_width=args.beam, epochs=args.epochs)
  File "train.py", line 471, in train
    plan = model.init(sql)
  File "/home/jae.sim/git/LOGER/model/dqn.py", line 135, in init
    return self.init(plan, grad=grad, return_graph=return_graph)
  File "/home/jae.sim/git/LOGER/model/dqn.py", line 138, in init
    graph : dgl.DGLHeteroGraph = self.model_step1(graph)
  File "/home/jae.sim/.conda/envs/loger/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/jae.sim/.conda/envs/loger/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/jae.sim/git/LOGER/model/step1.py", line 220, in forward
    table_x = self.table_transform(g)
  File "/home/jae.sim/.conda/envs/loger/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/jae.sim/.conda/envs/loger/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/jae.sim/git/LOGER/model/step1.py", line 70, in forward
    table_others = self.schema_prepare(table_others)
  File "/home/jae.sim/.conda/envs/loger/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/jae.sim/.conda/envs/loger/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/jae.sim/.conda/envs/loger/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 117, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: long int != float
```
{{% /details %}}

6) epoche 변경
```python
config.py
    #epochs = 200
    epochs = 300
```
7) api 맞추기
```python
graph_transformer_layer.py
        #g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        #g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))
```
8) timer 추가 및 balsa 와 비슷하게 맞추기
```python
    _epoch_timer = timer()  #< -추가 
    use_beam = beam_width >= 1

    test_latency = pd.DataFrame()
    expert_latency = pd.DataFrame()

    with _epoch_timer:  #< -추가 후 아래 block 감싸기
...
### 아래 블록 추가

                        # Test, Expert
                        pivot_df = df.pivot_table(columns='filename', values='raw_cost')
                        pivot_df['epoch_time'] = _epoch_timer.cur_time
                        test_latency = pd.concat([test_latency, pivot_df], ignore_index = True)
                        test_latency.to_csv('results/test_latency.csv', index=False)

                        pivot_df = df.pivot_table(index=None, columns='filename', values='raw_origin')
                        expert_latency = pd.concat([expert_latency, pivot_df], ignore_index = True)
                        expert_latency.to_csv('results/expert_latency.csv', index=False)

###
                        log('Resampling')

```