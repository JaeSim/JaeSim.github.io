+++
date = '2025-07-04T13:26:47+09:00'
subtitle =  'HANA DB에 Leanred Query Optimizer 솔루션들을 포팅하기 위한 분석'
weight = 6
title = 'HANA Database Porting'
tags = ["DBMS", "Database", "Optimizer", "Learned Query Optimizer", "Reinforcement Learning", "HANA database"]
categories = ["Learned Query Optimizer"]
+++

# **Learned Query Optimizer 포팅을 위한 HANA DB의 분석 내용**

## **SAP trial**
1. SAP BTP trial 링크
   - https://www.sap.com/products/technology-platform/trial.html
2. BTP trial 생성 및 BTP cockpit 접속
   - ID = P2009440130
   - 이후 스텝은 https://account.hanatrial.ondemand.com/trial/#/home/trial 관련 내용 따름
3. trial account 생성 - sub account 생성 및 sub account 클릭
    - ***!! 주의 !!***   US region으로 생성해야 HANA Cloud DB 서비스를 사용할 수 있음
4. Service -> Service MarketPlace 에서 HANA Cloud tools 구독 생성
    - create SAP HANA Cloud - plan: tools 생성
5. Security User에 Role assign
    - Security -> Users -> 본인 계정 선택 -> Role Collections -> Assign Role Collection -> `SAP HANA` keyword 및 `cloud` keyword 모두 체크하여 반영
6. SAP HANA Cloud 생성
    - Service -> Instances and Subscription -> SAP HANA Cloud 생성 클릭
7. SAP HANA Instance 생성
    - allow all ip address 를 설정
8. 생성된 instance의 connections의 SQL Endpoint가 host
    - instance를 클릭후 ... 에가면 open in SAP HANA Cockpit 이나 다른 Database Explorer 등으로 볼 수 있다
- 참고 아래 instance 스펙으로 고정됨. trial은 총 2개의 instance를 지원하는 듯함
  - Memory   16 GB
  - Storage   80 GB
  - Compute   1 vCPUs

- 아래 내용은 Cloud trial 이어서 HANA DB Cloud를 테스트 할수 있지만 좀더 제한적임. BTP trial이 좀더 많은 기능을 포함하고 있다.
   - https://www.sap.com/products/data-cloud/hana/trial.html
   - 위 사이트에서 트라이얼 시작(30일) 으로 별도의 User ID와 Password 를 메모
   - re-register trial을 하지 않으면 database안의 내용물은 유지되는 것으로 확인 ()기간동안..)

### **HANA DB Cloud와 python 연결을 위한 세팅**
#### **Driver 및 CLI 세팅 다운로드**
 - https://help.sap.com/docs/SAP_HANA_CLIENT/f1b440ded6144a54ada97ff95dac7adf/39eca89d94ca464ca52385ad50fc7dea.html?locale=en-US 참조
 - `HXEDownloadManager_linux.bin` 실행 -> Client 다운로드  (clients_linux_x86_64.tgz )
 ```sh
 ./HXEDownloadManager_linux.bin
 tar -vzxf clients_linux_x86_64.tgz
 tar -vzxf hdb_client_linux_x86_64.tgz
 cd HDB_CLIENT_LINUX_X86_64
 ./hdbsetup
 ```
 - hdb_client까지 설치하면 설치한 경로안에 `hdbcli-N.N.N.tar.gz` 가 있음. 이것을 설치
```sh
pip install /{path}/hdbcli-N.N.N.tar.gz
```
 
#### **sample code**
 - 아래 python 코드로 연동이 가능하나 hostname은 cockpit 가 아닌 database의 hostname을 쓰도록 주의
 ```python
from hdbcli import dbapi
import contextlib


@contextlib.contextmanager
def __connect():
    conn = dbapi.connect(address=<hostname>, port=443, user=<user>, password=<password>)
    try:
        cursor = conn.cursor()
        yield (cursor, conn)
    finally :
        cursor.close()
        conn.close()

def ExecuteSql(cursor, sql):
    try:
        cursor.execute(sql)
        result_sets = []
        while True:
            rows = cursor.fetchall()
            result_sets.append(rows)
            if not cursor.nextset():
                break
    except Exception as e:
        print('commdb error:', e)
    return result_sets[0]

if __name__ == "__main__":
    sql = "select * from GX_EMPLOYEES limit 10";
    with __connect() as (cursor, conn):
        result = ExecuteSql(cursor, sql)
        print(result)
 ```
 - hostname은 trial 기준으로 SAP HANA Database Explorer 에서 database 우클릭 -> properties 에 있는 host 를 입력. user 및 password는 trail 시작시 얻었던 것을 입력
 
#### **cli로 접근하기**
  - 앞서 얻은 호스트, user, password를 이용해서 hanadb on-premise를 설치할때 얻은 hdbsql로 접근이 가능
  ```sh
  sudo /usr/sap/HXE/HDB90/exe/hdbsql -n <hostname>:443 -u <user> -p <Password>
  ```

## **HANA DB on-premise 설치**
### **Trial 다운로드**
- 평가판(trial) 다운로드 위치 : https://www.sap.com/products/data-cloud/hana/express-trial.html
- 관련 참조 페이지
    - https://pages.community.sap.com/topics/hana
    - https://tools.hana.ondemand.com/#hanatools
    - https://gist.github.com/tlinnet/4196fd9c7aa43e37ea71e69bdc40a7dd
    - https://developers.sap.com/tutorial-navigator.html?search=install&tag=software-product%3Atechnology-platform%2Fsap-hana

### **설치 커맨드**
 - JAVA_HOME export as version 8
 ```sh
 vim ~/.bashrc
 # 하단에 아레 정보 입력
 export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

 source ~/.bashrc
 ```
 - 설치 파일 실행 권한 부여
```sh
chmod +x HXEDownloadManager_linux.bin
./HXEDownloadManager_linux.bin
```
 - server only installer 선택<br>
  binary installer -> [ server only installer , Clients (Linux x86/64) 두개 선택 ] 
 - 압축 해제 및 실행
```sh
tar -xvzf hxe.tgz

sudo mv /bin/sh /bin/sh.orig
sudo ln -s /bin/bash /bin/sh

sudo setup_hxe.sh
```

 - sudo 권한을 주지 않고 하려면 아래 수행을 해야하는데, 차라리 sudo 권한을 주는것을 권장. ㅔ
```sh
vim setupo_hxe.sh
  // checkRootuser 주석
  // checkBash 주석
Enter SAP HANA system ID [HXE]:

Enter instance number [90]:

Enter local host name [bdai-ThinkStation-P920]:
```
 - password 설정 (SAP**** 로 설정함)
 ```
 - 8 or more letters
 - At least 1 uppercase letter
 - At least 1 lowercase letter
 - At least 1 number
 ```
 -  우분투 설치하려면 아래 가이드 따를것 <br>
https://gist.github.com/tlinnet/4196fd9c7aa43e37ea71e69bdc40a7dd
 - 가이드에 있는 내용
 ```sh
sudo cp /usr/sbin/update-rc.d /usr/sbin/update-rc.d.orig
sudo sed -i '/^        usage("unknown option");/i \        \if (/^-n$/) { next }' /usr/sbin/update-rc.d
 ```
 - password 설정
 - 실행을 통해서 확인 (아래커맨드로 접속가능)
 ```sh

#export /usr/sap/<SID>/HDB<instance>/exe
export PATH=/usr/sap/HXE/HDB90/exe:$PATH
source ~/.bashrc
# ./hdbsql -n localhost:39013 -u System -p {Password}

sudo /usr/sap/HXE/HDB90/exe/hdbsql -n localhost:39013 -u System -p {Password}
 ```

 - HANA_EXPRESS_20\hxe_info.tx 에서 아래 정보 확인가능
 ```vim
  1 [hxe_version]
  2 HANA, express edition=2.00.082.00.20250528.1
  3 #
  4 HDB_AFL=2.00.082.00
  5 HDB_LCM=2.00.082.00
  6 HDB_SERVER=2.00.082.00
  7 HANA_CLIENT=2.24.19
  8 #
  9 XSA_RT=1.3.7
10 XSAC_COCKPIT=2.18.0
11 XSAC_HRTT=2.16.250401
12 XSAC_SAP_WEB_IDE=4.8.2
13 XSAC_PORTAL_SERV=2.8.0
14 XSAC_SERVICES=1.7.31
15 XSAC_UI5_FESV6=1.71.68
16 XSAC_UI5_FESV9=1.108.31
17 XSAC_XSA_COCKPIT=1.1.44
18 #
19 [hxe_installation]
20 INSTALL_TYPE=Binary
21 INSTALL_DATE=
~
 ```

## **동작 내용**
### **explain 관련 short 테스트**
 - TEST1, TEST2, TEST3 테이블을 만들고 임시 data를 넣고, EXPLAIN을 했을때 응답을 출력해본 것
{{% details title="SQL 수행" open=false %}}
```sh
hdbsql SYSTEMDB=> SELECT TABLE_NAME  FROM M_TABLES  WHERE SCHEMA_NAME = CURRENT_SCHEMA;
TABLE_NAME
"TEST1"
"TEST2"
"TEST3"

CREATE COLUMN TABLE TEST1 ( id VARCHAR(10), text VARCHAR(10) );

CREATE COLUMN TABLE TEST2 ( id VARCHAR(10), content VARCHAR(10) );

CREATE COLUMN TABLE TEST3 ( id VARCHAR(10), comment VARCHAR(10) );

CREATE INDEX idx_test3_id ON TEST3(id);


INSERT INTO TEST1 (id, text) VALUES ('A001', 'Text1');  -- TEST2, TEST3에도 존재
INSERT INTO TEST1 (id, text) VALUES ('A002', 'Text2');  -- TEST2, TEST3에도 존재
INSERT INTO TEST1 (id, text) VALUES ('A003', 'Text3');  -- TEST2, TEST3에도 존재
INSERT INTO TEST1 (id, text) VALUES ('A005', 'Text5');  -- TEST2에는 있지만 TEST3에는 없음
INSERT INTO TEST1 (id, text) VALUES ('A007', 'Text7');  -- TEST2에는 있지만 TEST3에는 없음
INSERT INTO TEST1 (id, text) VALUES ('A009', 'Text9');  -- TEST3에는 있지만 TEST2에는 없음
INSERT INTO TEST1 (id, text) VALUES ('A011', 'Text11'); -- TEST2, TEST3에는 없는 새로운 ID
INSERT INTO TEST1 (id, text) VALUES ('A012', 'Text12'); -- TEST2, TEST3에는 없는 새로운 ID


INSERT INTO TEST2 (id, content) VALUES ('A001', 'c1');
INSERT INTO TEST2 (id, content) VALUES ('A002', 'c2');
INSERT INTO TEST2 (id, content) VALUES ('A003', 'c3');
INSERT INTO TEST2 (id, content) VALUES ('A004', 'c4');
INSERT INTO TEST2 (id, content) VALUES ('A005', 'c5');
INSERT INTO TEST2 (id, content) VALUES ('A006', 'c6'); -- TEST3에 없는 ID
INSERT INTO TEST2 (id, content) VALUES ('A007', 'c7'); -- TEST3에 없는 ID

INSERT INTO TEST3 (id, comment) VALUES ('A001', 'p1');
INSERT INTO TEST3 (id, comment) VALUES ('A002', 'p2');
INSERT INTO TEST3 (id, comment) VALUES ('A003', 'p3');
INSERT INTO TEST3 (id, comment) VALUES ('A008', 'p8'); -- TEST2에 없는 ID
INSERT INTO TEST3 (id, comment) VALUES ('A009', 'p9'); -- TEST2에 없는 ID
INSERT INTO TEST3 (id, comment) VALUES ('A010', 'p10'); -- TEST2에 없는 ID


SELECT T1.id, T1.text, T2.content, T3.comment FROM TEST1 T1 INNER JOIN TEST2 T2 ON T1.id = T2.id INNER JOIN TEST3 T3 ON T1.id = T3.id;


hdbsql SYSTEMDB=> SELECT T1.id, T1.text, T2.content, T3.comment FROM TEST1 T1 INNER JOIN TEST2 T2 ON T
1.id = T2.id INNER JOIN TEST3 T3 ON T1.id = T3.id;
ID,TEXT,CONTENT,COMMENT
"A001","Text1","c1","p1"
"A002","Text2","c2","p2"
"A003","Text3","c3","p3"
3 rows selected (overall time 35.804 msec; server time 29.875 msec)



EXPLAIN PLAN SET STATEMENT_NAME = 'ORIG' FOR SELECT T1.id, T1.text, T3.comment FROM TEST1 T1 INNER JOIN TEST3 T3 ON T1.id = T3.id;
SELECT * FROM EXPLAIN_PLAN_TABLE ORDER BY TIMESTAMP DESC;

STATEMENT_NAME,OPERATOR_NAME,OPERATOR_DETAILS,OPERATOR_PROPERTIES,EXECUTION_ENGINE,DATABASE_NAME,SCHEM
A_NAME,TABLE_NAME,TABLE_TYPE,TABLE_SIZE,OUTPUT_SIZE,SUBTREE_COST,OPERATOR_ID,PARENT_OPERATOR_ID,LEVEL,
POSITION,HOST,PORT,TIMESTAMP,CONNECTION_ID

"ORIG","PROJECT","T1.ID, T1.TEXT, T3.COMMENT","PHYSICAL_ENUM_BY: HEX_PROJECT","HEX",?,?,?,?,?                            ,4,0.000000781081778845808,1,?          ,1,1,"bdai-thinkstation-p920",39001,"2025-03-20 12:23:54.078000000",117755
"ORIG","  INDEX JOIN","INDEX JOIN CONDITION: T1.ID = T3.ID","PHYSICAL_ENUM_BY: HEX_INDEX_JOIN","HEX","SYSTEMDB","SYSTEM","TEST3","COLUMN TABLE",6,4,0.000000781081778845808,2,1,2,1,"bdai-thinkstation-p920",39001,"2025-03-20 12:23:54.078000000",117755
"ORIG","    TABLE SCAN","","PHYSICAL_ENUM_BY: HEX_TABLE_SCAN","HEX","SYSTEMDB","SYSTEM","TEST1","COLUMN TABLE",8,8,0.000000024,3,2,3,1,"bdai-thinkstation-p920"
,39001,"2025-03-20 12:23:54.078000000",117755



EXPLAIN PLAN SET STATEMENT_NAME = 'FORCE_HSJO' FOR SELECT T1.id, T1.text, T3.comment FROM TEST1 T1 INNER JOIN TEST3 T3 ON T1.id = T3.id WITH HINT( HASH_JOIN );
4 rows selected (overall time 10.147 msec; server time 4031 usec)


"FORCE_HSJO","PROJECT","T1.ID, T1.TEXT, T3.COMMENT","ATTACHED HINT LIST (HASH_JOIN), PHYSICAL_ENUM_BY: HEX_PROJECT","HEX",?,?,?,?,? ,20,0.000003031119839811333,1,?                  ,1,1,"bdai-thinkstation-p920",39001,"2025-03-20 12:36:14.827000000",118106
"FORCE_HSJO","  INDEX JOIN","INDEX JOIN CONDITION: T1.ID = T3.ID","TRANSLATION TABLE (LEFT): T1.ID = T3.ID, PHYSICAL_ENUM_BY: HEX_INDEX_JOIN","HEX","SYSTEMDB","SYSTEM","TEST3","COLUMN TABLE",22,20,0.000003031119839811333,2,1,2,1,"bdai-thinkstation-p920",39001,"2025-03-20 12:36:14.827000000",118106
"FORCE_HSJO","    TABLE SCAN","","PHYSICAL_ENUM_BY: HEX_TABLE_SCAN","HEX","SYSTEMDB","SYSTEM","TEST1","COLUMN TABLE",29,29,0.000000087,3,2,3,1,"bdai-thinkstat
ion-p920",39001,"2025-03-20 12:36:14.827000000",118106
```

{{% /details %}}



## **dataset을 IMPORT를 하기 위해서 수정해야 했던 것들**
 - 어느것이 정확하게 동작하는지 기억x
```sql
ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM')  SET ('import_export', 'csv_import_path_filter') = '/home/jae.sim/hanadb/datasets/job/'  WITH RECONFIGURE;
ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM') set ('import', 'enable_csv_import_path_filter') = 'true' with reconfigure;
SELECT FILE_NAME, SECTION, KEY, VALUE FROM M_INIFILE_CONTENTS WHERE SECTION = 'import'  AND KEY = 'csv_import_path_filter';
SELECT FILE_NAME, SECTION, KEY, VALUE FROM M_INIFILE_CONTENTS WHERE SECTION = 'import_export'  ;
alter system alter configuration ( 'indexserver.ini','SYSTEM' ) set ( 'import_export','enable_csv_import_path_filter' ) = 'false' with reconfigure
ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM') SET ('import', 'csv_import_path_filter') = '/home/jae.sim/hanadb/datasets/job/'  WITH RECONFIGURE;
```

## **HANA DB 지원**

### **EXPLAIN and ANALYZE 지원 관련**
- EXPLAIN의 경우 `EXPLAIN PLAN` keyword로 지원
  - https://help.sap.com/docs/SAP_HANA_PLATFORM/bed8c14f9f024763b0777aa72b5436f6/c0d42fd3bb571014a0688254f3de593f.html
  - Join order 와 physical Op를 확인 가능
- ***ANALYZE의 경우 SQL Level에서는 미지원***
  - `M_SQL_PLAN_CACHE` 로 수행한 결과를 분석해야함
  - 해당 cache안에 join order등 어떠한 정보들을 볼 수 있는지
{{% details title="제공되는 정보" open=false %}}
  **TOTAL_EXECUTION_TIME** 을 제공. 그러나 sub-tree에 대한 정보는 제공하지 않음.
  ```
  HOST	PORT	VOLUME_ID	STATEMENT_STRING	STATEMENT_HASH	USER_NAME	SESSION_USER_NAME
  SCHEMA_NAME	SESSION_PROPERTIES	IS_VALID	LAST_INVALIDATION_REASON	IS_INTERNAL	IS_DISTRIBUTED_EXECUTION
  COMPILATION_OPTIONS	IS_PINNED_PLAN	PINNED_PLAN_ID	ABAP_VARCHAR_MODE	APPLICATION_NAME	APPLICATION_SOURCE
  ACCESSED_TABLES	ACCESSED_TABLE_NAMES	ACCESSED_OBJECTS	ACCESSED_OBJECT_NAMES	TABLE_LOCATIONS	TABLE_TYPES
  EXECUTION_ENGINE	HEX_REJECTION_REASON	PLAN_SHARING_TYPE	OWNER_CONNECTION_ID	PLAN_ID	PLAN_MEMORY_SIZE
  REFERENCE_COUNT	PARAMETER_COUNT	UPDATED_TABLE_OID	LOGICAL_CONNECTION_VOLUME_ID	EXECUTION_COUNT
  EXECUTION_COUNT_BY_ROUTING	PREFERRED_ROUTING_VOLUME_IDS	TOTAL_CURSOR_DURATION	AVG_CURSOR_DURATION
  MIN_CURSOR_DURATION	MAX_CURSOR_DURATION	TOTAL_EXECUTION_TIME	AVG_EXECUTION_TIME	MIN_EXECUTION_TIME
  MAX_EXECUTION_TIME	TOTAL_EXECUTION_OPEN_TIME	AVG_EXECUTION_OPEN_TIME	MIN_EXECUTION_OPEN_TIME	MAX_EXECUTION_OPEN_TIME	
  TOTAL_EXECUTION_FETCH_TIME	AVG_EXECUTION_FETCH_TIME	MIN_EXECUTION_FETCH_TIME	MAX_EXECUTION_FETCH_TIME	
  TOTAL_EXECUTION_CLOSE_TIME	AVG_EXECUTION_CLOSE_TIME	MIN_EXECUTION_CLOSE_TIME	MAX_EXECUTION_CLOSE_TIME
  TOTAL_METADATA_CACHE_MISS_COUNT	TOTAL_TABLE_LOAD_TIME_DURING_PREPARATION	AVG_TABLE_LOAD_TIME_DURING_PREPARATION	
  MIN_TABLE_LOAD_TIME_DURING_PREPARATION	MAX_TABLE_LOAD_TIME_DURING_PREPARATION	PREPARATION_COUNT
  TOTAL_PREPARATION_TIME	AVG_PREPARATION_TIME	MIN_PREPARATION_TIME	MAX_PREPARATION_TIME
  TOTAL_RESULT_RECORD_COUNT	TOTAL_LOCK_WAIT_COUNT	TOTAL_LOCK_WAIT_DURATION	LAST_CONNECTION_ID
  LAST_EXECUTION_TIMESTAMP	LAST_PREPARATION_TIMESTAMP	TOTAL_EXECUTION_MEMORY_SIZE	AVG_EXECUTION_MEMORY_SIZE	
  MIN_EXECUTION_MEMORY_SIZE	MAX_EXECUTION_MEMORY_SIZE	TOTAL_EXECUTION_CPU_TIME	AVG_EXECUTION_CPU_TIME
  MIN_EXECUTION_CPU_TIME	MAX_EXECUTION_CPU_TIME	AVG_SERVICE_NETWORK_REQUEST_COUNT	MAX_SERVICE_NETWORK_REQUEST_COUNT
  TOTAL_SERVICE_NETWORK_REQUEST_COUNT	AVG_CALLED_THREAD_COUNT	MAX_CALLED_THREAD_COUNT	TOTAL_CALLED_THREAD_COUNT
  TOTAL_BATCH_EXECUTION_COUNT	AVG_BATCH_EXECUTION_COUNT	MIN_BATCH_EXECUTION_COUNT	MAX_BATCH_EXECUTION_COUNT
  AVG_SERVICE_NETWORK_REQUEST_DURATION	MAX_SERVICE_NETWORK_REQUEST_DURATION	TOTAL_SERVICE_NETWORK_REQUEST_DURATION
  AVG_SERVICE_NETWORK_REQUEST_SIZE	MAX_SERVICE_NETWORK_REQUEST_SIZE	TOTAL_SERVICE_NETWORK_REQUEST_SIZE
  TOTAL_BUFFER_CACHE_PAGE_HIT_COUNT	AVG_BUFFER_CACHE_PAGE_HIT_COUNT	MIN_BUFFER_CACHE_PAGE_HIT_COUNT	MAX_BUFFER_CACHE_PAGE_HIT_COUNT
  TOTAL_BUFFER_CACHE_PAGE_MISS_COUNT	AVG_BUFFER_CACHE_PAGE_MISS_COUNT	MIN_BUFFER_CACHE_PAGE_MISS_COUNT
  MAX_BUFFER_CACHE_PAGE_MISS_COUNT	TOTAL_BUFFER_CACHE_IO_READ_SIZE	AVG_BUFFER_CACHE_IO_READ_SIZE
  MIN_BUFFER_CACHE_IO_READ_SIZE	MAX_BUFFER_CACHE_IO_READ_SIZE	TOTAL_BUFFER_CACHE_PINNED_MEMORY_SIZE
  AVG_BUFFER_CACHE_PINNED_MEMORY_SIZE	MIN_BUFFER_CACHE_PINNED_MEMORY_SIZE
  ```
{{% /details %}}

### **Timeout Setting**
 - Python
  ```python
  from hdbcli import dbapi
  
  conn = dbapi.connect(address='hana_host', port=30015, user='user', password='pw')
  cursor = conn.cursor()
  cursor.execute("SET STATEMENT_TIMEOUT = 30000")  # 30초
  cursor.execute("SELECT ...")
  ```
 - database level
  ```sql
  SET STATEMENT_TIMEOUT = 60000;  -- 60초 
  ```

### **hanadb join에 대한 키워드**
```text
HEX_HASH_JOIN
Guides the optimizer to prefer HEX hash joins over other joins.

NO_HEX_HASH_JOIN
Guides the optimizer to avoid HEX hash joins.

HEX_INDEX_JOIN
Guides the optimizer to prefer HEX index joins over other joins.

NO_HEX_INDEX_JOIN
Guides the optimizer to avoid HEX index joins.

HEX_NESTED_LOOP_JOIN
Guides the optimizer to prefer HEX nested loop joins over other joins.

NO_HEX_NESTED_LOOP_JOIN
Guides the optimizer to avoid HEX nested loop joins.

CONCAT_FILTER
Guides the optimizer to prefer HEX concat replacements.

NO_CONCAT_FILTER
Guides the optimizer to avoid HEX concat replacements.

HEX_RANGE_JOIN
Guides the optimizer to prefer HEX range joins over other joins.

NO_HEX_RANGE_JOIN
Guides the optimizer to avoid HEX range joins.

HEX_HASHED_RANGE_JOIN
Guides the optimizer to prefer HEX hashed range joins over other joins.

NO_HEX_HASHED_RANGE_JOIN
Guides the optimizer to avoid HEX hashed range joins.

HEX_TABLE_SCAN
Guides the optimizer to prefer HEX table scans over unique index searches.

NO_HEX_TABLE_SCAN
Guides the optimizer to avoid HEX table scans.

HEX_UNIQUE_INDEX_SEARCH
Guides the optimizer to prefer HEX unique index searches over table scans.

NO_HEX_UNIQUE_INDEX_SEARCH
Guides the optimizer to avoid HEX unique index searches.

HEX_LIMIT
Guides the optimizer to prefer HEX limits over top K sorts.

NO_HEX_LIMIT
Guides the optimizer to avoid HEX limits.

HEX_TOPK_SORT
Guides the optimizer to prefer HEX top K sorts over HEX limits.

NO_HEX_TOPK_SORT
Guides the optimizer to avoid HEX top K sorts.
```


### **HANA DB 로그 관련 위치**
 - 아래 위치에 각각의 core log가 나옴
```sh
      [1] /hana/shared/HXE/HDB90/bdai-thinkstation-p920/log/*
      [2] /hana/shared/HXE/HDB90/bdai-thinkstation-p920/log/grmg/*
      [3] /hana/shared/HXE/HDB90/bdai-thinkstation-p920/trace/*
      [4] /hana/shared/HXE/profile/HXE_HDB90_bdai-thinkstation-p920
      [5] /hana/shared/HXE/profile/*

-rw-r----- 1 hxeadm sapsys  1359872  3월 21 11:04 logsegment_000_directory.dat
(base) jae.sim@bdai-ThinkStation-P920:/hana/shared/log$ sudo vim HXE/mnt00001/hdb00001/logsegment_000_directory.dat

0:/hana/shared/HXE$ sudo tail -F HDB90/bdai-thinkstation-p920/trace/daemon__0144876__children.trc
```
 - log trace 관련 정보
     - https://help.sap.com/docs/SAP_HANA_PLATFORM/6b94445c94ae495c83a19646e7c3fd56/7e31247372fb4dd7b8c6bbac758b8c91.html <br>
       SQL trace	Collect information about all SQL statements executed on the index server	SAP HANA database explorer


## **쿼리 샘플들**
 - 실험용 데이터 적재, 및 clear 용 쿼리 모음집
{{% details title="SQL 샘플" open=false %}}
```sql
ALTER SYSTEM ALTER CONFIGURATION ('nameserver.ini', 'SYSTEM')  SET ('import_export', 'csv_import_path_filter') = '/home/jae.sim/hanadb/datasets/job/'  WITH RECONFIGURE;


SELECT 'aka_name' AS table_name, COUNT(*) AS row_count FROM aka_name;
SELECT 'aka_title' AS table_name, COUNT(*) AS row_count FROM aka_title;
SELECT 'cast_info' AS table_name, COUNT(*) AS row_count FROM cast_info;
SELECT 'char_name' AS table_name, COUNT(*) AS row_count FROM char_name;
SELECT 'comp_cast_type' AS table_name, COUNT(*) AS row_count FROM comp_cast_type;
SELECT 'company_name' AS table_name, COUNT(*) AS row_count FROM company_name;
SELECT 'company_type' AS table_name, COUNT(*) AS row_count FROM company_type;
SELECT 'complete_cast' AS table_name, COUNT(*) AS row_count FROM complete_cast;
SELECT 'info_type' AS table_name, COUNT(*) AS row_count FROM info_type;
SELECT 'keyword' AS table_name, COUNT(*) AS row_count FROM keyword;
SELECT 'kind_type' AS table_name, COUNT(*) AS row_count FROM kind_type;
SELECT 'link_type' AS table_name, COUNT(*) AS row_count FROM link_type;
SELECT 'movie_companies' AS table_name, COUNT(*) AS row_count FROM movie_companies;
SELECT 'movie_info_idx' AS table_name, COUNT(*) AS row_count FROM movie_info_idx;
SELECT 'movie_keyword' AS table_name, COUNT(*) AS row_count FROM movie_keyword;
SELECT 'movie_link' AS table_name, COUNT(*) AS row_count FROM movie_link;
SELECT 'name' AS table_name, COUNT(*) AS row_count FROM name;
SELECT 'role_type' AS table_name, COUNT(*) AS row_count FROM role_type;
SELECT 'title' AS table_name, COUNT(*) AS row_count FROM title;
SELECT 'movie_info' AS table_name, COUNT(*) AS row_count FROM movie_info;
SELECT 'person_info' AS table_name, COUNT(*) AS row_count FROM person_info;


truncate table aka_name;
truncate table aka_title;
truncate table cast_info;
truncate table char_name;
truncate table comp_cast_type;
truncate table company_name;
truncate table company_type;
truncate table complete_cast;
truncate table info_type;
truncate table keyword;
truncate table kind_type;
truncate table link_type;
truncate table movie_companies;
truncate table movie_info_idx;
truncate table movie_keyword;
truncate table movie_link;
truncate table name;
truncate table role_type;
truncate table title;
truncate table movie_info;
truncate table person_info;


insert into aka_name values(1,4061927,"Smithl",55,S5325,J2542,S53,25c9d464e3ff2957533546aa92b397ed);


CREATE TABLE aka_name1 (id integer NOT NULL PRIMARY KEY,person_id integer NOT NULL,name nvchar(50));

insert into aka_name1 values(1,4061927,'Smithl');
```
{{% /details %}}


## **SAP HANA studio 다운로드**
 - HANA DB는 SQL 분석을 위해서 `SAP HANA studio` 라는 것을 지원함
 - 다운로드 위치 
    - https://help.sap.com/docs/SAP_Commissions_K8s/021635a4731e40f4a2784f4613e632d2/bd2eec401f7c47cf9d2dbed0d0b24233.html
    - https://launchpad.support.sap.com/#/softwarecenter/



## **SSB workload 업로드**
### **SSB Dataset 생성**
 - ssb 데이터 생성
```sh
./dbgen -s 10 -T c    # customer
./dbgen -s 10 -T s    # supplier
./dbgen -s 10 -T p    # part
./dbgen -s 10 -T d    # date
./dbgen -s 10 -T l    # lineorder
```

### **table 생성**
 - table load 용 sql 준비 및 실행
    - 주요특징 : balsa 프로젝트에 있는 table.sql 항목에서 `TEXT`를 `CLOB` 으로 변환
    - hdbsql 이용할때 -I {filename} 옵션으로 사용
    - NVARCHAR를 사용 (guided from SAP team).
```SQL
DO
BEGIN
    IF EXISTS (SELECT * FROM TABLES WHERE TABLE_NAME = 'LINEORDER') THEN
        EXEC 'DROP TABLE LINEORDER';
    END IF;

    IF EXISTS (SELECT * FROM TABLES WHERE TABLE_NAME = 'DATE') THEN
        EXEC 'DROP TABLE "DATE"';
    END IF;

    IF EXISTS (SELECT * FROM TABLES WHERE TABLE_NAME = 'PART') THEN
        EXEC 'DROP TABLE PART';
    END IF;

    IF EXISTS (SELECT * FROM TABLES WHERE TABLE_NAME = 'SUPPLIER') THEN
        EXEC 'DROP TABLE SUPPLIER';
    END IF;

    IF EXISTS (SELECT * FROM TABLES WHERE TABLE_NAME = 'CUSTOMER') THEN
        EXEC 'DROP TABLE CUSTOMER';
    END IF;
END;

--
-- customer
--
CREATE TABLE customer (
  c_custkey    INTEGER  PRIMARY KEY,
  c_name       CLOB,
  c_address    CLOB,
  c_city       NVARCHAR(10),
  c_nation     NVARCHAR(15),
  c_region     NVARCHAR(12),
  c_phone      NVARCHAR(15),
  c_mktsegment NVARCHAR(10),
  dummy        CLOB-- dbgen last delimiter
);

--
-- date
--
CREATE TABLE date (
  d_datekey          DATE PRIMARY KEY,
  d_date             NVARCHAR(18),
  d_dayofweek        NVARCHAR(9),
  d_month            NVARCHAR(9),
  d_year             INTEGER,
  d_yearmonthnum     INTEGER,
  d_yearmonth        NVARCHAR(7),
  d_daynuminweek     INTEGER,
  d_daynuminmonth    INTEGER,
  d_daynuminyear     INTEGER,
  d_monthnuminyear   INTEGER,
  d_weeknuminyear    INTEGER,
  d_sellingseason    CLOB,
  d_lastdayinweekfl  NVARCHAR(1),
  d_lastdayinmonthfl NVARCHAR(1),
  d_holidayfl        NVARCHAR(1),
  d_weekdayfl        NVARCHAR(1),
  dummy              CLOB -- dbgen last delimiter
);

--
-- part
--
CREATE TABLE part (
  p_partkey   INTEGER PRIMARY KEY,
  p_name      CLOB,
  p_mfgr      NVARCHAR(6),
  p_category  NVARCHAR(7),
  p_brand1    NVARCHAR(9),
  p_color     NVARCHAR(11),
  p_type      CLOB,
  p_size      INTEGER,
  p_container NVARCHAR(10),
  dummy       CLOB  -- dbgen last delimiter
);

--
-- supplier
--
CREATE TABLE supplier (
  s_suppkey INTEGER PRIMARY KEY,
  s_name    NVARCHAR(25),
  s_address CLOB,
  s_city    NVARCHAR(10),
  s_nation  NVARCHAR(15),
  s_region  NVARCHAR(12),
  s_phone   NVARCHAR(15),
  dummy            CLOB -- dbgen last delimiter
);

--
-- lineorder
--
CREATE TABLE lineorder (
  lo_orderkey      BIGINT, -- Consider SF 300+
  lo_linenumber    INTEGER,
  lo_custkey       INTEGER, -- FK to C_CUSTKEY
  lo_partkey       INTEGER, -- FK to P_PARTKEY
  lo_suppkey       INTEGER, -- FK to S_SUPPKEY
  lo_orderdate     DATE,    -- FK to D_DATEKEY
  lo_orderpriority NVARCHAR(15),
  lo_shippriority  NVARCHAR(1),
  lo_quantity      INTEGER,
  lo_extendedprice NUMERIC,
  lo_ordtotalprice NUMERIC,
  lo_discount      NUMERIC,
  lo_revenue       NUMERIC,
  lo_supplycost    NUMERIC,
  lo_tax           NUMERIC,
  lo_commitdate    DATE, -- FK to D_DATEKEY
  lo_shipmod       NVARCHAR(10),
  dummy            CLOB, -- dbgen last delimiter
  CONSTRAINT lo_pkey  PRIMARY KEY(lo_orderkey, lo_linenumber),
  FOREIGN KEY (lo_custkey)  REFERENCES customer (c_custkey),
  FOREIGN KEY (lo_partkey)  REFERENCES part (p_partkey),
  FOREIGN KEY (lo_suppkey)  REFERENCES supplier (s_suppkey),
  FOREIGN KEY (lo_orderdate)  REFERENCES date (d_datekey)
);
```
 - `select * from "REFERENTIAL_CONSTRAINTS"` 로 foreign key가 등록된것을 확인 할 수 있음

### **data load**
 - explorer를 통해서 csv로 업로드 할 수 있으나, 향후 on-premise에서 업로드를 생각하여 python + hdbcli로 업로드하는 것으로 기술
```python

# 👉 테이블 정의: 각 테이블별 컬럼 이름 (dummy 제외)
table_columns = {
    "customer": ["c_custkey", "c_name", "c_address", "c_city", "c_nation", "c_region", "c_phone", "c_mktsegment"],
    "supplier": ["s_suppkey", "s_name", "s_address", "s_city", "s_nation", "s_region", "s_phone"],
    "part": ["p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1", "p_color", "p_type", "p_size", "p_container"],
    "date": ["d_datekey", "d_date", "d_dayofweek", "d_month", "d_year", "d_yearmonthnum", "d_yearmonth",
             "d_daynuminweek", "d_daynuminmonth", "d_daynuminyear", "d_monthnuminyear", "d_weeknuminyear",
             "d_sellingseason", "d_lastdayinweekfl", "d_lastdayinmonthfl", "d_holidayfl", "d_weekdayfl"],
    "lineorder": ["lo_orderkey", "lo_linenumber", "lo_custkey", "lo_partkey", "lo_suppkey", "lo_orderdate",
                  "lo_orderpriority", "lo_shippriority", "lo_quantity", "lo_extendedprice", "lo_ordtotalprice",
                  "lo_discount", "lo_revenue", "lo_supplycost", "lo_tax", "lo_commitdate", "lo_shipmod"]
}

# 👉 파일명 리스트 (확장자 .tbl)
tbl_files = ["customer.tbl", "supplier.tbl", "part.tbl", "date.tbl", "lineorder.tbl"]

for tbl_file in tbl_files:
    table_name = os.path.splitext(tbl_file)[0]
    columns = table_columns.get(table_name)

    if not columns:
        print(f"❌ Unknown table: {table_name}")
        continue

    print(f"🚀 Loading data into table: {table_name.upper()} ...")

    file_path = os.path.join("hana", tbl_file)
    batch = []
    batch_size = 5000
    inserted = 0

    placeholders = ','.join(['?'] * len(columns))
    col_str = ', '.join(columns)
    query = f'INSERT INTO "GE211441"."{table_name.upper()}" ({col_str}) VALUES ({placeholders})'

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split('|')
            if len(fields) < len(columns):
                continue
            values = fields[:len(columns)]
            batch.append(values)

            if len(batch) >= batch_size:
                try:
                    cursor.executemany(query, batch)
                    conn.commit()
                    inserted += len(batch)
                    print(f"✅ Inserted {inserted} rows so far into {table_name.upper()}")
                    batch = []
                except dbapi.Error as e:
                    print(f"⚠️ Error in batch insert: {e}")
                    batch = []

    # 마지막 남은 batch 처리
    if batch:
        try:
            cursor.executemany(query, batch)
            conn.commit()
            inserted += len(batch)
            print(f"✅ Inserted total {inserted} rows into {table_name.upper()}")
        except dbapi.Error as e:
            print(f"⚠️ Error in final batch insert: {e}")

cursor.close()
conn.close()
```
 - row 업로드 확인
 ```sql
 SELECT 'CUSTOMER' AS TABLE_NAME, COUNT(*) AS ROW_COUNT FROM "GE211441"."CUSTOMER"
UNION ALL
SELECT 'SUPPLIER', COUNT(*) FROM "GE211441"."SUPPLIER"
UNION ALL
SELECT 'PART', COUNT(*) FROM "GE211441"."PART"
UNION ALL
SELECT 'DATE', COUNT(*) FROM "GE211441"."DATE"
UNION ALL
SELECT 'LINEORDER', COUNT(*) FROM "GE211441"."LINEORDER";

TABLE_NAME,ROW_COUNT
CUSTOMER,300000
SUPPLIER,20000
PART,800000
DATE,2556
LINEORDER,59986052
```


## **JOB workload 업로드**
### **JOB Dataset**
 - imdb.tgz download
 - tar -vzxf imdb.tgz

### **JOB table create**
 - imdb.tgz에 첨부되어있던, schematext.sql을 아래 python으로 변환
 - 되도록 NVARCHAR를 사용하나, 5000 사이즈 제한으로 CLOB을 사용을 사용함
```python
import re
from pathlib import Path

def convert_varchar_to_nvarchar(sql_text: str) -> str:
    # 1. character varying(N) -> NVARCHAR(N)
    sql_text = re.sub(
        r'character\s+varying\s*\(\s*(\d+)\s*\)',
        r'NVARCHAR(\1)',
        sql_text,
        flags=re.IGNORECASE,
    )

    # 2. character varying -> NVARCHAR(255)
    sql_text = re.sub(
        r'character\s+varying\b',
        r'CLOB',
        sql_text,
        flags=re.IGNORECASE,
    )

    return sql_text

if __name__ == "__main__":
    # 예시: input.sql 파일 읽어서 변환 후 output.sql로 저장
    src_path = Path("schematext.sql")
    dst_path = Path("output_schematext.sql")

    sql_src = src_path.read_text(encoding="utf-8")
    sql_out = convert_varchar_to_nvarchar(sql_src)
    dst_path.write_text(sql_out, encoding="utf-8")

    print(f"✅ 변환 완료: {dst_path}")
```
 - 해당 sql을 이용하여 table 생성
 - fkadd.sql을 업로드 (from balsa)

### **JOB data bulk upload**
 - imdb.tgz로 파생된 csv파일들을 아래 python으로 업로드
