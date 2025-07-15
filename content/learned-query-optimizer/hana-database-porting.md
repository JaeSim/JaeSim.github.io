+++
date = '2025-07-04T13:26:47+09:00'
subtitle =  'HANA DB에 Leanred Query Optimizer 솔루션들을 포팅하기 위한 분석'
weight = 6
title = 'HANA Database Porting'
tags = ["DBMS", "Database", "Optimizer", "Learned Query Optimizer", "Reinforcement Learning", "HANA database"]
categories = ["Learned Query Optimizer"]
+++

# **Learned Query Optimizer 포팅을 위한 HANA DB의 분석 내용**
## **설치**
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
