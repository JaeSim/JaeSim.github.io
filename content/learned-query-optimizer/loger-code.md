+++
date = '2025-06-12T11:50:33+09:00'
title = 'LOGER 코드 분석'
subtitle =  'LOGER 프로젝트의 코드 분석 내용을 담은 내용입니다.'
weight = 4
tags = ["DBMS", "Database", "Optimizer", "Learned Query Optimizer", "Reinforcement Learning", "LOGER"]
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
python train.py --database ssb --port 5437 --host /tmp -U "" --dataset dataset/ssb_train dataset/ssb_test

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

## **High level flow**

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


## **내부 특성**

### **LOGER는 Join Order 와 Join Op만 설정**
- _Scan Op 에 대한 학습은 진행하지 않음_

### **NO_USE_HASH, NO_USE_MERGE 등이 ROSS 개념을 이용하여 학습**
 - 실제 학습이 되어서 호출되는 hint 들을 보면, NO_USE_HASH, NO_USE_MERGE, NO_USE_NL 을 사용.
 - 이는 논문에 언급된 ROSS(Restricted Operator Search Space) 의 구현.

### **train_mode, test_mode of model**
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

### **workload sql 파싱 위치**
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

### **SQL parse and composition 위치**
 - 처음 train, test workload의 path를 읽으면, 이후 Sql class로 변환하는데 (위에 언급된 파서를 이용), 이것은 각 요소를 분해해서 각기 요소로 가지고 있음(e.g., SelectStmt, fromClause, whereClause...)
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
 - plan.py에도 동일하게 분해해서 값들이 있고, 학습을 통해서 각각의 요소의 (join order, join method) 구성을 바꾼다. <br>
   그리고 _str 할때 이러한 각 요소를 다시 조립해서 hint + sql 을 만든다. (DBMS에 호출할 최종 SQL)
 ```python
 # LOGER/core/plan.py
 class Plan:
    ...
     def __reset(self):
        self.root_nodes = set(self.sql.aliases)
        self.parent = {}
        self.direct_parent = {}
        self.left_children = {}
        self.right_children = {}
        self.total_branch_nodes = 0
        self.join_on_left = {}
        self.join_on_right = {}
        self.children_table_aliases = {}
    ...
    def __str__(self):
        hints = []
        self._hint_str(hints)
        hints = f'/*+ {" ".join(hints)} */ '
 ```


### **상용 db 연결의 흔적**
- oracle database를 사용한것으로 추정.
- oracle database에 모두 동작하지는 않고, postgres 먼저 구현뒤 oracle을 구현한 것으로 추정.
- **mssql 도 oracle db 포팅 코드를 기반으로 작성**
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

## **주요 구현 이슈**
### **NO_USE_HASH, NO_USE_MERGE 등이 MSSQL에 없음**
- LOGER의 restrict operation 은 postgre 에서는 지원이 되나, oracle, MSSQL 은 NO_* Op가 없음
- 지원해야할 Op
  ```python
    ALL_JOIN = 0
    NO_NEST_LOOP_JOIN = 1
    NO_MERGE_JOIN = 2
    NO_HASH_JOIN = 4
    NEST_LOOP_JOIN = 6
    MERGE_JOIN = 5
    HASH_JOIN = 3

    # postgre
    @classmethod
    def str_join_method(cls, type):
        if type == cls.NO_NEST_LOOP_JOIN:
            return "NoNestLoop"
        elif type == cls.NO_MERGE_JOIN:
            return "NoMergeJoin"
        elif type == cls.NO_HASH_JOIN:
            return "NoHashJoin"
        elif type == cls.NEST_LOOP_JOIN:
            return "NestLoop"
        elif type == cls.MERGE_JOIN:
            return "MergeJoin"
        elif type == cls.HASH_JOIN:
            return "HashJoin"
        return None

    # oracle
    @classmethod
    def oracle_join_method(cls, type):
        if type == cls.NO_NEST_LOOP_JOIN:
            return "NO_USE_NL"
        elif type == cls.NO_MERGE_JOIN:
            return "NO_USE_MERGE"
        elif type == cls.NO_HASH_JOIN:
            return "NO_USE_HASH"
        elif type == cls.NEST_LOOP_JOIN:
            return "USE_NL"
        elif type == cls.MERGE_JOIN:
            return "USE_MERGE"
        elif type == cls.HASH_JOIN:
            return "USE_HASH"
        return None

    # mssql
    @classmethod
    def mssql_join_method(cls, type):
        if type == cls.NO_NEST_LOOP_JOIN:
            return ???
        elif type == cls.NO_MERGE_JOIN:
            return ???
        elif type == cls.NO_HASH_JOIN:
            return ???
        elif type == cls.NEST_LOOP_JOIN:
            return "USE_NL"
        elif type == cls.MERGE_JOIN:
            return "USE_MERGE"
        elif type == cls.HASH_JOIN:
            return "USE_HASH"
        return None
  ```
 - postgresql = pg_hint_plan  : https://github.com/ossc-db/pg_hint_plan/blob/master/pg_hint_plan.c
 - oracle docu
 - hanadb   https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/hint-details#loio4ba9edce1f2347a0b9fcda99879c17a1__hints_for_controlling_join1
```
INDEX_JOIN
Guides the optimizer to join input relations through index searches.

NO_INDEX_JOIN
Guides the optimizer to avoid joining the input relations through index searches.

HASH_JOIN
Guides the optimizer to join the input relations through probing the hash table.

NO_HASH_JOIN
Guides the optimizer to avoid joining the input relations through probing the hash table
```

- mssql 일 경우 중간에 adaptive layer를 두는 방식으로 추가 구현
     1) 별도의 keyword를 주입하고,  
     2) execute 할때 두번 나뉘어 호출 (예. NO_USE_NL 이라면 HASH 와 MERGE를 주입한 query를 실행) 
     3) 이중 더 나은 execution time을 cost로 사용.
    - **이는 향후 논문에서 추가 기술이 필요**
    -  __latency 와 cost 안에서 구현하기 vs __execute 안에서 구현하기
    ```python
    # _mssql_db.py 안의
    ...
    def __latency(self, sql, cache=True, detail=False):
    with Timer() as timer:
        res = self.__execute(sql)
    ...
    def cost(self, sql, cache=True):
    assert self.__db is not None
    if cache and sql in self.__cost_cache:
        return self.__cost_cache[sql]

    # self.__execute(f"explain plan set statement_id = 'current' for {sql}")
    with Cursor(timeout= self.statement_timeout / 1000) as (wrapped_cursor, conn):
        cur = wrapped_cursor
        res = ExplainSql(cur, sql)
        #cost = self.first_element(f"select cost from plan_table where statement_id = 'current'")
        cost = float(res.split("cost=")[1].split("..")[1].split(" ")[0])
        self.__db.commit()
        self.__cost_cache[sql] = cost
        self.__auto_save()
        return cost
    ...
    ```

### **Node 구조가 Balsa와 다름**
- 발사의 경우 graph 구조 형태로 Node를 가지고 있으며, 이것들이 각기 하나의 Scan Op 또는 Join Op를 가지고 있음. 이 Node들을 조합하면서, 하나의 Hint를 만들어내는데,
- LOGER의 경우 Plan.py 안에 node 들이 _배열_ 형태로 있고, 이것들을 좌우로 참조해내가는 식으로 합쳐나감
- 따라서, Balsa에서 코드 내부에서 (Node로 부터) Hint를 만드는 기존 방식과 다르게, LOGER에서는 만들어진 

### **상용 db을 위한 postgre 세팅**
**LOGER는 oracle로 테스트 한것으로 보이는데, 독특한점은 oracle로 train을 시작해도 postgre에 동일한 dataset이 ready되어 있어야한다.** <br>
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

## **mssql 포팅을 위한 작업일지**

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
9) mssql_hint용 힌트 만들기 로직을, 기존 postgre 와 동일하게 만든 뒤에 (mssql은 이해하지 못함)<br>
  이후 추가적은 sql을 execution 하는 방식으로 구현<br>
     - **NO_HASH, NO_MERGE 등이 MSSQL에 없음** 항목을 우회하기 위함
     - 예시) A, B, C를 조인할때 , USE_ALL (NO_USE_NL(A,B), C) 이라고 가정하면 <br>
     SQL 쿼리를 총 2번 호출하여 우선 cost가 낮은 값으로 기록 <br>
        1) USE_ALL (USE_HASH(A,B), C)
        2) USE_ALL (USE_MERGE(A,B), C)
     - 모호한 JOIN( NO_USE* ) 의 갯수만큼 {{< katex display=false >}}2^n{{< /katex >}} 개의 쿼리 호출이 필요
     - `reassemblage_sql` method 구현
   ```python
    def __latency(self, sql, cache=True, detail=False):
        modified_sqls, need_duplicated_execute = self.reassemblage_sql(sql)

        best_cost = 0
        best_res = None
        if need_duplicated_execute :
            for i, query in enumerate(modified_sqls):
                cost, res = self.__latency_inner(query, cache, detail)
                if best_res == None : 
                    best_cost = cost
                    best_res = res
                else :
                    if cost < best_cost :
                        best_cost = cost
                        best_res = res

            if best_cost == 0 : # error case
                raise NotImplementedError
            
        else : 
            best_cost, best_res = self.__latency_inner(modified_sqls, cache, detail)

        self.__executed[sql] = best_res
        return best_cost
   ```
   - 이를 위해서, postgres hint에서 join 되어야할 target table 구분을 하도록 수정 (, 로 구분) <br>
   (e.g., /*+ NO_USE_MERGE(s , lo) NO_USE_MERGE(lo s , d) NO_USE_NL(d lo s , p) */ )
   ```python
   # LOGER/core/plan.py
    def __mssql_hint(self, node, hints : list):
        ...
            if join_method is not None:
                hints.append(f'{join_method}({" ".join(sorted(self._descendants[left_alias]))} , {" ".join(sorted(self._descendants[right_alias]))})')
        ...
   ```
   - 이후 reassemble_sql 에서 , 기준으로 파싱해서 재조립