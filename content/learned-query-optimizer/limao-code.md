+++
date = '2025-08-28T11:32:49+09:00'
title = 'Limao 코드 분석'
subtitle =  'LIMAO 프로젝트의 코드 분석 내용을 담은 내용입니다.'
weight = 7
tags = ["DBMS", "Database", "Optimizer", "Learned Query Optimizer", "Reinforcement Learning", "LIMAO"]
categories = ["Learned Query Optimizer"]
+++

# **LIMAO 프로젝트의 코드 분석 내용**

## **기본정보**
- LIMAO: A Framework for Lifelong Modular Learned Query Optimization
- github repository
https://github.com/Tsihan/LIMAOLifeLongRLDB
- ***balsa 프로젝트 위에서 코딩으로 구현되어 있는데,, 썩 잘코딩되진...***
- github branch가 main은 사실상 original balsa project이며 branch를 바꿔서 테스트해야함
- model.py, optimizer.py, run.py 이 3개를 주로 코딩 작업 한것으로 이해
- plan decomposition(join을 breakpoint) 과 replay(episode당 하나씩 호출), k-prototype(서브모듈 클러스터링용), switch 시나리오 실험이 key idea 라고 이해

### **LIMAO 실행을 위한 셋업**
#### **basic 셋업**
 - conda create and git clone
``` sh
conda create -n limao python=3.8
conda activate limao

git clone https://github.com/Tsihan/LIMAOLifeLongRLDB.git

cd LIMAOLifeLongRLDB

git checkout origin/final_switching_workload
git checkout -b final_switching_workload

pip install -e .
pip install -e pg_executor
pip install -r requirements.txt
pip install six
pip install numpy==1.20.3
pip install pandas
pip install pyyaml
pip install --upgrade "protobuf==3.20.3"
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install click
pip install wandb
pip install knodes
pip install -U "pydantic>=2" "ray>=2.7" "wandb>=0.16"

```
#### **postgresql 셋업 & pg_hint_plan**
 - https://github.com/Tsihan/LIMAOLifeLongRLDB Readme.md 파일 참조

#### **JOB setup**
 - prepare dataset.
 ```sh
mkdir -p /mydata/datasets/job && cd /mydata/datasets/job
wget -c http://homepages.cwi.nl/~boncz/job/imdb.tgz
tar -xvzf imdb.tgz
 ```
 - prepend header   (balsa 것 그대로 사용)
 ```sh
 python3 /mydata/LIMAOLifeLongRLDB/scripts/prepend_imdb_headers.py
 flags.DEFINE_string('csv_dir', '/mydata/datasets/job', 'Directory to IMDB CSVs.')
 ```
 - start postgre    ( balsa의 postgre conf를 base로 `work_mem 4GB -> 8GB`, `listner_address * -> 'localhost'`, `max_wal_size 1GB -> 10GB` 로 변경)
 ```sh
pg_ctl -D /mydata/databases initdb
cp /mydata/LIMAOLifeLongRLDB/conf/balsa-postgresql.conf /mydata/databases/postgresql.conf
pg_ctl -D /mydata/databases start -l logfile
 ```

 - 포트설정을 위해서 아래내용 수정후 pg_execution 재 install (pg_execution.py)
 ```python
 LOCAL_DSN = "host=/tmp dbname=imdbload port=5437"
 ```
  ```python
 pip install -e pg_executor
 ```
#### **SSB setup**
 - https://github.com/nuko-yokohama/ssb-postgres/tree/master
```sh
cd ssb-dbgen
make

./dbgen -s 10 -T c    # customer
./dbgen -s 10 -T s    # supplier
./dbgen -s 10 -T p    # part
./dbgen -s 10 -T d    # date
./dbgen -s 10 -T l    # lineorder

createdb ssb -p 5437

psql ssb -f table.sql -p 5437
psql ssb -f load.sql -p 5437
```


#### **tpcds setup**
 - https://github.com/gregrahn/tpcds-kit
```sh
cd tools
make CC=gcc-9 OS=LINUX
```
```sh
createdb -p 5437 tpcds
psql -d tpcds -f tpcds.sql -p 5437

mkdir data
./dsdgen -SCALE 4 -DIR data
```
 - 파싱 문제가 있어서, 아래 명령어를 입력해주어야 한다.
 ```sh
 cd data
 sed -i 's/|$//' *.dat
 ```

 psql -p 5437 -d tpcds -f <data/load.sql가 있는 경로>
```sql
TRUNCATE catalog_returns; TRUNCATE store; TRUNCATE dbgen_version; TRUNCATE item; TRUNCATE web_page; TRUNCATE promotion;
TRUNCATE inventory; TRUNCATE web_sales; TRUNCATE store_sales; TRUNCATE catalog_sales; TRUNCATE call_center; TRUNCATE income_band; TRUNCATE date_dim;
TRUNCATE reason; TRUNCATE store_returns; TRUNCATE web_returns; TRUNCATE customer_demographics; TRUNCATE warehouse; TRUNCATE ship_mode;
TRUNCATE customer_address; TRUNCATE time_dim; TRUNCATE household_demographics; TRUNCATE web_site; TRUNCATE customer; TRUNCATE catalog_page;

COPY catalog_returns FROM '/{path}/tpcds-kit/tools/data/catalog_returns.dat' DELIMITER '|' NULL '';
COPY store FROM '/{path}/tpcds-kit/tools/data/store.dat' DELIMITER '|' NULL '';
COPY dbgen_version FROM '/{path}/tpcds-kit/tools/data/dbgen_version.dat' DELIMITER '|' NULL '';
COPY item FROM '/{path}/tpcds-kit/tools/data/item.dat' DELIMITER '|' NULL '';
COPY web_page FROM '/{path}/tpcds-kit/tools/data/web_page.dat' DELIMITER '|' NULL '';
COPY promotion FROM '/{path}/tpcds-kit/tools/data/promotion.dat' DELIMITER '|' NULL '';
COPY inventory FROM '/{path}/tpcds-kit/tools/data/inventory.dat' DELIMITER '|' NULL '';
COPY web_sales FROM '/{path}/tpcds-kit/tools/data/web_sales.dat' DELIMITER '|' NULL '';
COPY store_sales FROM '/{path}/tpcds-kit/tools/data/store_sales.dat' DELIMITER '|' NULL '';
COPY catalog_sales FROM '/{path}/tpcds-kit/tools/data/catalog_sales.dat' DELIMITER '|' NULL '';
COPY call_center FROM '/{path}/tpcds-kit/tools/data/call_center.dat' DELIMITER '|' NULL '';
COPY income_band FROM '/{path}/tpcds-kit/tools/data/income_band.dat' DELIMITER '|' NULL '';
COPY date_dim FROM '/{path}/tpcds-kit/tools/data/date_dim.dat' DELIMITER '|' NULL '';
COPY reason FROM '/{path}/tpcds-kit/tools/data/reason.dat' DELIMITER '|' NULL '';
COPY store_returns FROM '/{path}/tpcds-kit/tools/data/store_returns.dat' DELIMITER '|' NULL '';
COPY web_returns FROM '/{path}/tpcds-kit/tools/data/web_returns.dat' DELIMITER '|' NULL '';
COPY customer_demographics FROM '/{path}/tpcds-kit/tools/data/customer_demographics.dat' DELIMITER '|' NULL '';
COPY warehouse FROM '/{path}/tpcds-kit/tools/data/warehouse.dat' DELIMITER '|' NULL '';
COPY ship_mode FROM '/{path}/tpcds-kit/tools/data/ship_mode.dat' DELIMITER '|' NULL '';
COPY customer_address FROM '/{path}/tpcds-kit/tools/data/customer_address.dat' DELIMITER '|' NULL '';
COPY time_dim FROM '/{path}/tpcds-kit/tools/data/time_dim.dat' DELIMITER '|' NULL '';
COPY household_demographics FROM '/{path}/tpcds-kit/tools/data/household_demographics.dat' DELIMITER '|' NULL '';
COPY web_site FROM '/{path}/tpcds-kit/tools/data/web_site.dat' DELIMITER '|' NULL '';
COPY customer FROM '/{path}/tpcds-kit/tools/data/customer.dat' DELIMITER '|' NULL '';
COPY catalog_page FROM '/{path}/tpcds-kit/tools/data/catalog_page.dat' DELIMITER '|' NULL '';
```
- 쿼리생성
```sh
# 쿼리 생성
`./dsqgen -DIRECTORY <query_templates 디렉토리 경로> -INPUT <query_templates/templates.lst 경로> -SCALE 10 -QUALIFY Y -VERBOSE Y -DIALECT netezza -OUTPUT_DIR <쿼리 저장할 경로 (아무데나)>`
```

 ### **실행중 에러**
 #### **numpy memory 할당 이슈**
 - numpy가 메모리를 할당 못하는 이슈
 ```log
 numpy.core._exceptions.MemoryError: Unable to allocate 5.29 GiB for an array with shape (524348, 22, 123) and data type float32
 ```
 - 아래커맨드로 현재 정책확인  0 = 기본 , 1= 허용, 2 = 엄격하게 제한
 ```sh
 cat /proc/sys/vm/overcommit_memory
 ```
- 아래 커맨드로 일시적으로 멤모리 할당 허용 (재부팅시 초기화)
 ```sh
 sudo sysctl -w vm.overcommit_memory=0
 ```

#### **FileNotFoundError 에러**
- file path가 안맞는 에러
```log
FileNotFoundError: [Errno 2] No such file or directory: '/mydata/LIMAOLifeLongRLDB/balsa/deal_assorted_text/indexes_env_matrix.txt
```
- 아래처럼 경로 변경  (k-prototype 용 magic 분류 라고 이해)
```python
    index_path='{your path}/LIMAOLifeLongRLDB/balsa/deal_assorted_text/indexes_env_matrix.txt',
    operator_path='{your path}/LIMAOLifeLongRLDB/balsa/deal_assorted_text/operators_env_matrix.txt',
    sql_path='{your path}/LIMAOLifeLongRLDB/balsa/deal_assorted_text/sql_feature_encode_matrix.txt',
    query_path='{your path}/LIMAOLifeLongRLDB/balsa/deal_assorted_text/query_enc_matrix.txt',
```