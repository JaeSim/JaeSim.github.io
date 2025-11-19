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

### **EXPLAIN 관련**
- 아래와 같이 총 2회(3회) 의 SQL을 형태로 호출하여 결과를 받아야하며 row 형태로 응답을 받을 수 있음
```SQL
DELETE FROM explain_plan_table WHERE statement_name = 'ORIG';

EXPLAIN PLAN SET STATEMENT_NAME = 'ORIG' FOR
select * from akatitle;

SELECT * FROM EXPLAIN_PLAN_TABLE WHERE STATEMENT_NAME = 'ORIG' 
```

### **ANALYZE 관련**
- 아래 형식처럼 target_sql을 감싸서 질의하면 xml 형태의 응답을 받을 수 있음 <br>
단, 권한을 가지고 있어야함 (BTP trial로 생성된 hana db의 경우 권한을 부여하고 있음)
```SQL

DO
BEGIN
    DECLARE lv_planviz_xml CLOB;
    CALL GET_PLANVIZ_EXECUTED_PLAN('
    select * from aka_title
    ', lv_planviz_xml);

    SELECT :lv_planviz_xml FROM DUMMY;
END;
```

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
--logical Enum rule
(NO_)AGGR_THRU_JOIN
(NO_)AGGR_THRU_FILTER
(NO_)PREAGGR_BEFORE_JOIN
(NO_)DOUBLE_PREAGGR_BEFORE_JOIN
(NO_)JOIN_THRU_JOIN
(NO_)JOIN_THRU_AGGR
(NO_)JOIN_THRU_UNION
(NO_)JOIN_THRU_FILTER
(NO_)DISTINCT_THRU_UNION
(NO_)PREAGGR_BEFORE_UNION
(NO_)DISJ_JOIN_INTO_UNION
(NO_)FILTER_THRU_JOIN
(NO_)FILTER_THRU_AGGR
(NO_)FILTER_THRU_UNION
(NO_)DOUBLE_JOIN_THRU_UNION_ALL

--join operator
(NO_)HEX_HASH_JOIN
(NO_)HEX_INDEX_JOIN
(NO_)HEX_RANGE_JOIN
(NO_)HEX_HASHED_RANGE_JOIN
(NO_)HEX_NESTED_LOOP_JOIN

--scan operator
(NO_)HEX_TABLE_SCAN
(NO_)HEX_INDEX_SCAN
(NO_)HEX_UNIQUE_INDEX_SEARCH

--index
(NO_)INDEX(your_table, 'index_name_to_avoid')
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

### **SSB table 생성**
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

### **SSB data load**
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
### **JOB Dataset 생성**
 - imdb.tgz download
 - tar -vzxf imdb.tgz

### **JOB table create**
 - imdb.tgz에 첨부되어있던, schematext.sql을 아래 python으로 변환
 ```sh
 wget -c https://event.cwi.nl/da/job/imdb.tgz && tar -xvzf imdb.tgz && popd
 ```
 - 되도록 NVARCHAR으로 사이즈 제한
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
        r'NVARCHAR(5000)',
        sql_text,
        flags=re.IGNORECASE,
    )

    return sql_text


def add_schema_to_create(sql_text: str, schema: str = "imdb") -> str:
    """
    CREATE TABLE table_name → CREATE TABLE schema.table_name
    이미 schema.table_name 인 경우는 건너뜀
    """
    pattern = re.compile(r'(\bCREATE\s+TABLE\s+)([^\s(]+)', flags=re.IGNORECASE)

    def repl(match):
        prefix, table_name = match.groups()
        if "." in table_name:  # 이미 schema 지정됨
            return match.group(0)
        return f"{prefix}{schema}.{table_name}"

    return pattern.sub(repl, sql_text)


if __name__ == "__main__":
    # 예시: input.sql 파일 읽어서 변환 후 output.sql로 저장
    src_path = Path("schematext.sql")
    dst_path = Path("output_schematext.sql")

    sql_src = src_path.read_text(encoding="utf-8")
    sql_out = convert_varchar_to_nvarchar(sql_src)
    sql_out = add_schema_to_create(sql_out)
    dst_path.write_text(sql_out, encoding="utf-8")

    print(f"✅ 변환 완료: {dst_path}")

```
 - 필요시에 hana explorer를 통해서 schema 생성
 ```SQL
 create schema imdb;
 ```
 - 해당 sql을 이용하여 table 생성
  ```sh
  sudo /usr/sap/HXE/HDB90/exe/hdbsql -n {host}:443 -u {user} -p {password} -I output_schematext.sql
  ```
 - fkindexex.sql을 업로드 (from balsa)  [필요시 imdb. 삭제]
```sql
create index company_id_movie_companies on imdb.movie_companies(company_id);
create index company_type_id_movie_companies on imdb.movie_companies(company_type_id);
create index info_type_id_movie_info_idx on imdb.movie_info_idx(info_type_id);
create index info_type_id_movie_info on imdb.movie_info(info_type_id);
create index info_type_id_person_info on imdb.person_info(info_type_id);
create index keyword_id_movie_keyword on imdb.movie_keyword(keyword_id);
create index kind_id_aka_title on imdb.aka_title(kind_id);
create index kind_id_title on imdb.title(kind_id);
create index linked_movie_id_movie_link on imdb.movie_link(linked_movie_id);
create index link_type_id_movie_link on imdb.movie_link(link_type_id);
create index movie_id_aka_title on imdb.aka_title(movie_id);
create index movie_id_cast_info on imdb.cast_info(movie_id);
create index movie_id_complete_cast on imdb.complete_cast(movie_id);
create index subject_id_complete_cast on imdb.complete_cast(subject_id);
create index status_id_complete_cast on imdb.complete_cast(status_id);
create index movie_id_movie_companies on imdb.movie_companies(movie_id);
create index movie_id_movie_info_idx on imdb.movie_info_idx(movie_id);
create index movie_id_movie_keyword on imdb.movie_keyword(movie_id);
create index movie_id_movie_link on imdb.movie_link(movie_id);
create index movie_id_movie_info on imdb.movie_info(movie_id);
create index person_id_aka_name on imdb.aka_name(person_id);
create index person_id_cast_info on imdb.cast_info(person_id);
create index person_id_person_info on imdb.person_info(person_id);
create index person_role_id_cast_info on imdb.cast_info(person_role_id);
create index role_id_cast_info on imdb.cast_info(role_id);
```
```sh
sudo /usr/sap/HXE/HDB90/exe/hdbsql -n {host}:443 -u {user} -p {password} -I fkindexex.sql
```
### **JOB data bulk upload**
 - imdb.tgz로 파생된 csv파일들을 아래 python으로 업로드
 - 아래 row들은 조금 다르게 업로드 될 수 있다는 점을 유의해야하고, title.csv 의 경우 수정이 필요하다
    - postgresql 은 copy로 업로드하는데, hanadb는 row 파일 -> pandas 로 load -> upload 과정에서 tokenizer 과정에 의해서 \ 처리가 다르게 된다는 점을 유의
```csv
# person_info.csv 의 아래 두 row는 \ 가 postgresql copy로 업로드하는 것과 다르게 업로드 될수 있는것을 유의해야한다
2671660,2604773,17,Daughter of Irish actor and raconteur 'Niall Toibin' (qv); \,
2671662,1562399,37,"\"The Sunday Times Culture\" (UK), 26 April 2009",
```
 - 다음 row는 수정필요
```csv
# title.csv
2522636,\Frag'ile\,,1,2010,,F624,,,,,c0b2e279bce6d3b1717e750a2591bb6d
# 다음과 같이 수정
2522636,\\Frag'ile\\,,1,2010,,F624,,,,,c0b2e279bce6d3b1717e750a2591bb6d
```

 - 5000자가 넘는 파일에 대해서는 짤라서 업로드
```python
import glob
import pandas as pd
from typing import Iterable, List, Any
import os, csv, tempfile
from hdbcli import dbapi
from sqlalchemy import create_engine

engine = create_engine(f'hana://{user}:{password}@{host}:443')

table_columns = {
    "aka_name": ["id", "person_id", "name", "imdb_index",
        "name_pcode_cf", "name_pcode_nf", "surname_pcode", "md5sum"],
    "aka_title": [
        "id", "movie_id", "title", "imdb_index", "kind_id", "production_year",
        "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "note", "md5sum"
    ],
    "cast_info": [
        "id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"
    ],
    "char_name": ["id", "name", "imdb_index", "imdb_id",
        "name_pcode_nf", "surname_pcode", "md5sum"],
    "comp_cast_type": ["id", "kind"],
    "company_name": ["id", "name", "country_code", "imdb_id",
        "name_pcode_nf", "name_pcode_sf", "md5sum"],
    "company_type": ["id", "kind"],
    "complete_cast": ["id", "movie_id", "subject_id", "status_id"],
    "info_type": ["id", "info"],
    "keyword": ["id", "keyword", "phonetic_code"],
    "kind_type": ["id", "kind"],
    "link_type": ["id", "link"],
    "movie_companies": ["id", "movie_id", "company_id", "company_type_id", "note"],
    "movie_info_idx": ["id", "movie_id", "info_type_id", "info", "note"],
    "movie_keyword": ["id", "movie_id", "keyword_id"],
    "movie_link": ["id", "movie_id", "linked_movie_id", "link_type_id"],
    "name": ["id", "name", "imdb_index", "imdb_id", "gender",
        "name_pcode_cf", "name_pcode_nf", "surname_pcode", "md5sum"],
    "role_type": ["id", "role"],
    "title": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id",
        "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"],
    "movie_info": ["id", "movie_id", "info_type_id", "info", "note"],
    "person_info": ["id", "person_id", "info_type_id", "info", "note"]
}

path = "./job_dataset"
#path = "./temp"
out_dir = os.path.join(path, "conv")
os.makedirs(out_dir, exist_ok=True)   # conv 폴더 없으면 생성
csv_files = glob.glob(os.path.join(path, "*.csv"))

# 방법 2: os.listdir 사용 (필터링)
csv_files = [
    os.path.join(path, f)
    for f in os.listdir(path)
    if f.lower().endswith(".csv")
]

fix_none = lambda x: None if (x is None or x == '') else ("None" if x == "None" else x)
MAX_NCHAR = 5000
for csv_file in csv_files:
    base = os.path.basename(csv_file)              # person_info.csv
    out_file = os.path.join(out_dir, base)         # ./temp/conv/person_info.csv

    out_file = csv_file
    #print(f"✅ {csv_file} → {out_file}")
    name, _ = os.path.splitext(out_file)
    table = os.path.basename(name)             # person_info
    print(table, out_file)
    read_kwargs = dict(sep=",", header=None, chunksize=500000, names=table_columns[table],
        engine="python", quotechar='"',
        doublequote=True,
        escapechar='\\',
        quoting=csv.QUOTE_MINIMAL,
        keep_default_na=False,
        na_filter=False,
        converters={col: fix_none for col in table_columns[table]}  # 모든 컬럼 적용
    )

    chunker = pd.read_csv(out_file, **read_kwargs)
    total_rows = 0
    for i, chunk in enumerate(chunker, start=1):
        obj_cols = chunk.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            # 유니코드 안전 슬라이싱(파이썬 슬라이스는 코드포인트 기준)
            for col in obj_cols:
                # None은 그대로 두고 문자열만 잘라줌
                # .str.slice는 NaN에 안전, keep_default_na=False 덕에 None만 존재
                chunk[col] = chunk[col].astype("object").map(
                    lambda v: (v[:MAX_NCHAR] if isinstance(v, str) and len(v) > MAX_NCHAR else v)
                )

        rows = len(chunk)
        total_rows += rows
        try:
            chunk.to_sql(schema='IMDB', name=table, con=engine, index=False, if_exists='append')
        except Exception as e:
            if not os.path.isdir("./error/"):
               os.mkdir("./error/")
            chunk.to_csv(f"./error/1.csv")
            with open(f'./error/error.log', 'w', encoding='utf-8') as f:
                f.write(str(e))
                f.close()
            raise NotImplementedError
        print(f"[{i:04d}] chunk rows={rows:,}  processed={total_rows:,}")

```
 - 후에 fk key add
```sh
sudo /usr/sap/HXE/HDB90/exe/hdbsql -n {host}:443 -u {user} -p {password} -I addfk.sql
```
 ```SQL
 ALTER TABLE "IMDB"."TITLE"
  ADD CONSTRAINT "FK_TITLE_KIND"
  FOREIGN KEY ("KIND_ID")
  REFERENCES "IMDB"."KIND_TYPE"("ID");

ALTER TABLE "IMDB"."AKA_NAME"
  ADD CONSTRAINT "PK_AKA_NAME_ID"
  PRIMARY KEY ("ID");  -- 이미 PK라면 생략

-- (주의) cast_info.person_id -> 'NAME'.ID 로 거는 것이 일반적입니다.
-- 질문에 있던 PG 에러는 aka_name 을 참조해서 난 것입니다.
ALTER TABLE "IMDB"."CAST_INFO"
  ADD CONSTRAINT "FK_CASTINFO_MOVIE"
  FOREIGN KEY ("MOVIE_ID")
  REFERENCES "IMDB"."TITLE"("ID");

ALTER TABLE "IMDB"."CAST_INFO"
  ADD CONSTRAINT "FK_CASTINFO_PERSON"
  FOREIGN KEY ("PERSON_ID")
  REFERENCES "IMDB"."NAME"("ID");

ALTER TABLE "IMDB"."CAST_INFO"
  ADD CONSTRAINT "FK_CASTINFO_PERSON_ROLE"
  FOREIGN KEY ("PERSON_ROLE_ID")
  REFERENCES "IMDB"."CHAR_NAME"("ID");

ALTER TABLE "IMDB"."CAST_INFO"
  ADD CONSTRAINT "FK_CASTINFO_ROLE"
  FOREIGN KEY ("ROLE_ID")
  REFERENCES "IMDB"."ROLE_TYPE"("ID");

ALTER TABLE "IMDB"."COMPLETE_CAST"
  ADD CONSTRAINT "FK_CC_MOVIE"
  FOREIGN KEY ("MOVIE_ID")
  REFERENCES "IMDB"."TITLE"("ID");

ALTER TABLE "IMDB"."COMPLETE_CAST"
  ADD CONSTRAINT "FK_CC_SUBJECT"
  FOREIGN KEY ("SUBJECT_ID")
  REFERENCES "IMDB"."COMP_CAST_TYPE"("ID");

ALTER TABLE "IMDB"."COMPLETE_CAST"
  ADD CONSTRAINT "FK_CC_STATUS"
  FOREIGN KEY ("STATUS_ID")
  REFERENCES "IMDB"."COMP_CAST_TYPE"("ID");

ALTER TABLE "IMDB"."MOVIE_COMPANIES"
  ADD CONSTRAINT "FK_MC_MOVIE"
  FOREIGN KEY ("MOVIE_ID")
  REFERENCES "IMDB"."TITLE"("ID");

ALTER TABLE "IMDB"."MOVIE_INFO"
  ADD CONSTRAINT "FK_MI_MOVIE"
  FOREIGN KEY ("MOVIE_ID")
  REFERENCES "IMDB"."TITLE"("ID");

ALTER TABLE "IMDB"."MOVIE_INFO"
  ADD CONSTRAINT "FK_MI_INFOTYPE"
  FOREIGN KEY ("INFO_TYPE_ID")
  REFERENCES "IMDB"."INFO_TYPE"("ID");

ALTER TABLE "IMDB"."MOVIE_INFO_IDX"
  ADD CONSTRAINT "FK_MII_MOVIE"
  FOREIGN KEY ("MOVIE_ID")
  REFERENCES "IMDB"."TITLE"("ID");

ALTER TABLE "IMDB"."MOVIE_INFO_IDX"
  ADD CONSTRAINT "FK_MII_INFOTYPE"
  FOREIGN KEY ("INFO_TYPE_ID")
  REFERENCES "IMDB"."INFO_TYPE"("ID");

ALTER TABLE "IMDB"."MOVIE_KEYWORD"
  ADD CONSTRAINT "FK_MK_MOVIE"
  FOREIGN KEY ("MOVIE_ID")
  REFERENCES "IMDB"."TITLE"("ID");

ALTER TABLE "IMDB"."MOVIE_KEYWORD"
  ADD CONSTRAINT "FK_MK_KEYWORD"
  FOREIGN KEY ("KEYWORD_ID")
  REFERENCES "IMDB"."KEYWORD"("ID");

ALTER TABLE "IMDB"."MOVIE_LINK"
  ADD CONSTRAINT "FK_ML_MOVIE"
  FOREIGN KEY ("MOVIE_ID")
  REFERENCES "IMDB"."TITLE"("ID");

ALTER TABLE "IMDB"."MOVIE_LINK"
  ADD CONSTRAINT "FK_ML_LINKTYPE"
  FOREIGN KEY ("LINK_TYPE_ID")
  REFERENCES "IMDB"."LINK_TYPE"("ID");

ALTER TABLE "IMDB"."PERSON_INFO"
  ADD CONSTRAINT "FK_PI_PERSON"
  FOREIGN KEY ("PERSON_ID")
  REFERENCES "IMDB"."NAME"("ID");

ALTER TABLE "IMDB"."PERSON_INFO"
  ADD CONSTRAINT "FK_PI_INFOTYPE"
  FOREIGN KEY ("INFO_TYPE_ID")
  REFERENCES "IMDB"."INFO_TYPE"("ID");
 ```

### **JOB 15x.sql query 변경**
 - 15b,c,d sql 에는 `at`라는 alias를 사용하나,  `at` 라는것이 hana db에서 키워드이므로 이슈가 발생
 - 15b,c,d에 at를 att로 변경 하여 실험

## **TPC-DS workload 업로드**

### **TPC-DS dataset 생성**
- git clone https://github.com/gregrahn/tpcds-kit
- tools build
```sh
cd tools
make CC=gcc-9 OS=LINUX
```
- create schema
```SQL
create schema tpcds
```
- 데이터 생성
```sh
./dsdgen -SCALE 4 -DIR data
```
 - 파싱 문제가 있어서, 아래 명령어를 입력해주어야 한다.
 ```sh
 sed -i 's/|$//' <data/*.dat (.dat 파일들이 있는 경로)>
 ```

### **TPC-DS table convert**
 - 아래 python으로 NVARCHAR로 변환

```python
from pathlib import Path
import re

def convert_varchar_to_nvarchar(sql_text: str) -> str:
    """
    CHAR(n), VARCHAR(n) → NVARCHAR(n) 변환
    """
    # VARCHAR 먼저 변환 (CHAR 안에 "VAR" 안겹치도록 순서 중요)
    sql_text = re.sub(r'\bVARCHAR\s*\((\d+)\)', r'NVARCHAR(\1)', sql_text, flags=re.IGNORECASE)
    # CHAR 변환
    sql_text = re.sub(r'\bCHAR\s*\((\d+)\)', r'NVARCHAR(\1)', sql_text, flags=re.IGNORECASE)
    return sql_text


def add_schema_to_create(sql_text: str, schema: str = "tpcds") -> str:
    """
    CREATE TABLE table_name → CREATE TABLE schema.table_name
    이미 schema.table_name 인 경우는 건너뜀
    """
    pattern = re.compile(r'(\bCREATE\s+TABLE\s+)([^\s(]+)', flags=re.IGNORECASE)

    def repl(match):
        prefix, table_name = match.groups()
        if "." in table_name:  # 이미 schema 지정됨
            return match.group(0)
        return f"{prefix}{schema}.{table_name}"

    return pattern.sub(repl, sql_text)

if __name__ == "__main__":
    src_path = Path("createtable.sql")
    dst_path = Path("output_createtable.sql")

    sql_src = src_path.read_text(encoding="utf-8")
    sql_out = convert_varchar_to_nvarchar(sql_src)
    sql_out = add_schema_to_create(sql_out, schema="tpcds")
    dst_path.write_text(sql_out, encoding="utf-8")

    print(f"✅ 변환 완료 (스키마=tpcds): {dst_path}")
```

### **TPC-DS data bulk upload**
 - 아래 스크립트로 업로드

```python
import glob
import pandas as pd
from typing import Iterable, List, Any
import os, csv, tempfile
from hdbcli import dbapi
from sqlalchemy import create_engine

engine = create_engine(f'hana://{user}:{password}@{host}:443')

table_columns = {
    "dbgen_version": [
        "dv_version", "dv_create_date", "dv_create_time", "dv_cmdline_args"
    ],
    "customer_address": [
        "ca_address_sk", "ca_address_id", "ca_street_number", "ca_street_name",
        "ca_street_type", "ca_suite_number", "ca_city", "ca_county", "ca_state",
        "ca_zip", "ca_country", "ca_gmt_offset", "ca_location_type"
    ],
    "customer_demographics": [
        "cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status",
        "cd_purchase_estimate", "cd_credit_rating", "cd_dep_count",
        "cd_dep_employed_count", "cd_dep_college_count"
    ],
    "date_dim": [
        "d_date_sk", "d_date_id", "d_date", "d_month_seq", "d_week_seq", "d_quarter_seq",
        "d_year", "d_dow", "d_moy", "d_dom", "d_qoy", "d_fy_year", "d_fy_quarter_seq",
        "d_fy_week_seq", "d_day_name", "d_quarter_name", "d_holiday", "d_weekend",
        "d_following_holiday", "d_first_dom", "d_last_dom", "d_same_day_ly", "d_same_day_lq",
        "d_current_day", "d_current_week", "d_current_month", "d_current_quarter", "d_current_year"
    ],
    "warehouse": [
        "w_warehouse_sk", "w_warehouse_id", "w_warehouse_name", "w_warehouse_sq_ft",
        "w_street_number", "w_street_name", "w_street_type", "w_suite_number",
        "w_city", "w_county", "w_state", "w_zip", "w_country", "w_gmt_offset"
    ],
    "ship_mode": [
        "sm_ship_mode_sk", "sm_ship_mode_id", "sm_type", "sm_code", "sm_carrier", "sm_contract"
    ],
    "time_dim": [
        "t_time_sk", "t_time_id", "t_time", "t_hour", "t_minute", "t_second",
        "t_am_pm", "t_shift", "t_sub_shift", "t_meal_time"
    ],
    "reason": [
        "r_reason_sk", "r_reason_id", "r_reason_desc"
    ],
    "income_band": [
        "ib_income_band_sk", "ib_lower_bound", "ib_upper_bound"
    ],
    "item": [
        "i_item_sk", "i_item_id", "i_rec_start_date", "i_rec_end_date", "i_item_desc",
        "i_current_price", "i_wholesale_cost", "i_brand_id", "i_brand", "i_class_id", "i_class",
        "i_category_id", "i_category", "i_manufact_id", "i_manufact", "i_size", "i_formulation",
        "i_color", "i_units", "i_container", "i_manager_id", "i_product_name"
    ],
    "store": [
        "s_store_sk", "s_store_id", "s_rec_start_date", "s_rec_end_date", "s_closed_date_sk",
        "s_store_name", "s_number_employees", "s_floor_space", "s_hours", "s_manager", "s_market_id",
        "s_geography_class", "s_market_desc", "s_market_manager", "s_division_id", "s_division_name",
        "s_company_id", "s_company_name", "s_street_number", "s_street_name", "s_street_type",
        "s_suite_number", "s_city", "s_county", "s_state", "s_zip", "s_country", "s_gmt_offset",
        "s_tax_precentage"
    ],
    "call_center": [
        "cc_call_center_sk", "cc_call_center_id", "cc_rec_start_date", "cc_rec_end_date",
        "cc_closed_date_sk", "cc_open_date_sk", "cc_name", "cc_class", "cc_employees", "cc_sq_ft",
        "cc_hours", "cc_manager", "cc_mkt_id", "cc_mkt_class", "cc_mkt_desc", "cc_market_manager",
        "cc_division", "cc_division_name", "cc_company", "cc_company_name", "cc_street_number",
        "cc_street_name", "cc_street_type", "cc_suite_number", "cc_city", "cc_county", "cc_state",
        "cc_zip", "cc_country", "cc_gmt_offset", "cc_tax_percentage"
    ],
    "customer": [
        "c_customer_sk", "c_customer_id", "c_current_cdemo_sk", "c_current_hdemo_sk",
        "c_current_addr_sk", "c_first_shipto_date_sk", "c_first_sales_date_sk", "c_salutation",
        "c_first_name", "c_last_name", "c_preferred_cust_flag", "c_birth_day", "c_birth_month",
        "c_birth_year", "c_birth_country", "c_login", "c_email_address", "c_last_review_date_sk"
    ],
    "web_site": [
        "web_site_sk", "web_site_id", "web_rec_start_date", "web_rec_end_date", "web_name",
        "web_open_date_sk", "web_close_date_sk", "web_class", "web_manager", "web_mkt_id",
        "web_mkt_class", "web_mkt_desc", "web_market_manager", "web_company_id", "web_company_name",
        "web_street_number", "web_street_name", "web_street_type", "web_suite_number", "web_city",
        "web_county", "web_state", "web_zip", "web_country", "web_gmt_offset", "web_tax_percentage"
    ],
    "store_returns": [
        "sr_returned_date_sk", "sr_return_time_sk", "sr_item_sk", "sr_customer_sk", "sr_cdemo_sk",
        "sr_hdemo_sk", "sr_addr_sk", "sr_store_sk", "sr_reason_sk", "sr_ticket_number",
        "sr_return_quantity", "sr_return_amt", "sr_return_tax", "sr_return_amt_inc_tax", "sr_fee",
        "sr_return_ship_cost", "sr_refunded_cash", "sr_reversed_charge", "sr_store_credit", "sr_net_loss"
    ],
    "household_demographics": [
        "hd_demo_sk", "hd_income_band_sk", "hd_buy_potential", "hd_dep_count", "hd_vehicle_count"
    ],
    "web_page": [
        "wp_web_page_sk", "wp_web_page_id", "wp_rec_start_date", "wp_rec_end_date",
        "wp_creation_date_sk", "wp_access_date_sk", "wp_autogen_flag", "wp_customer_sk",
        "wp_url", "wp_type", "wp_char_count", "wp_link_count", "wp_image_count", "wp_max_ad_count"
    ],
    "promotion": [
        "p_promo_sk", "p_promo_id", "p_start_date_sk", "p_end_date_sk", "p_item_sk", "p_cost",
        "p_response_target", "p_promo_name", "p_channel_dmail", "p_channel_email", "p_channel_catalog",
        "p_channel_tv", "p_channel_radio", "p_channel_press", "p_channel_event", "p_channel_demo",
        "p_channel_details", "p_purpose", "p_discount_active"
    ],
    "catalog_page": [
        "cp_catalog_page_sk", "cp_catalog_page_id", "cp_start_date_sk", "cp_end_date_sk",
        "cp_department", "cp_catalog_number", "cp_catalog_page_number", "cp_description", "cp_type"
    ],
    "inventory": [
        "inv_date_sk", "inv_item_sk", "inv_warehouse_sk", "inv_quantity_on_hand"
    ],
    "catalog_returns": [
        "cr_returned_date_sk", "cr_returned_time_sk", "cr_item_sk", "cr_refunded_customer_sk",
        "cr_refunded_cdemo_sk", "cr_refunded_hdemo_sk", "cr_refunded_addr_sk", "cr_returning_customer_sk",
        "cr_returning_cdemo_sk", "cr_returning_hdemo_sk", "cr_returning_addr_sk", "cr_call_center_sk",
        "cr_catalog_page_sk", "cr_ship_mode_sk", "cr_warehouse_sk", "cr_reason_sk", "cr_order_number",
        "cr_return_quantity", "cr_return_amount", "cr_return_tax", "cr_return_amt_inc_tax", "cr_fee",
        "cr_return_ship_cost", "cr_refunded_cash", "cr_reversed_charge", "cr_store_credit", "cr_net_loss"
    ],
    "web_returns": [
        "wr_returned_date_sk", "wr_returned_time_sk", "wr_item_sk", "wr_refunded_customer_sk",
        "wr_refunded_cdemo_sk", "wr_refunded_hdemo_sk", "wr_refunded_addr_sk", "wr_returning_customer_sk",
        "wr_returning_cdemo_sk", "wr_returning_hdemo_sk", "wr_returning_addr_sk", "wr_web_page_sk",
        "wr_reason_sk", "wr_order_number", "wr_return_quantity", "wr_return_amt", "wr_return_tax",
        "wr_return_amt_inc_tax", "wr_fee", "wr_return_ship_cost", "wr_refunded_cash",
        "wr_reversed_charge", "wr_account_credit", "wr_net_loss"
    ],
    "web_sales": [
        "ws_sold_date_sk", "ws_sold_time_sk", "ws_ship_date_sk", "ws_item_sk", "ws_bill_customer_sk",
        "ws_bill_cdemo_sk", "ws_bill_hdemo_sk", "ws_bill_addr_sk", "ws_ship_customer_sk", "ws_ship_cdemo_sk",
        "ws_ship_hdemo_sk", "ws_ship_addr_sk", "ws_web_page_sk", "ws_web_site_sk", "ws_ship_mode_sk",
        "ws_warehouse_sk", "ws_promo_sk", "ws_order_number", "ws_quantity", "ws_wholesale_cost",
        "ws_list_price", "ws_sales_price", "ws_ext_discount_amt", "ws_ext_sales_price", "ws_ext_wholesale_cost",
        "ws_ext_list_price", "ws_ext_tax", "ws_coupon_amt", "ws_ext_ship_cost", "ws_net_paid",
        "ws_net_paid_inc_tax", "ws_net_paid_inc_ship", "ws_net_paid_inc_ship_tax", "ws_net_profit"
    ],
    "catalog_sales": [
        "cs_sold_date_sk", "cs_sold_time_sk", "cs_ship_date_sk", "cs_bill_customer_sk", "cs_bill_cdemo_sk",
        "cs_bill_hdemo_sk", "cs_bill_addr_sk", "cs_ship_customer_sk", "cs_ship_cdemo_sk", "cs_ship_hdemo_sk",
        "cs_ship_addr_sk", "cs_call_center_sk", "cs_catalog_page_sk", "cs_ship_mode_sk", "cs_warehouse_sk",
        "cs_item_sk", "cs_promo_sk", "cs_order_number", "cs_quantity", "cs_wholesale_cost", "cs_list_price",
        "cs_sales_price", "cs_ext_discount_amt", "cs_ext_sales_price", "cs_ext_wholesale_cost",
        "cs_ext_list_price", "cs_ext_tax", "cs_coupon_amt", "cs_ext_ship_cost", "cs_net_paid",
        "cs_net_paid_inc_tax", "cs_net_paid_inc_ship", "cs_net_paid_inc_ship_tax", "cs_net_profit"
    ],
    "store_sales": [
        "ss_sold_date_sk", "ss_sold_time_sk", "ss_item_sk", "ss_customer_sk", "ss_cdemo_sk",
        "ss_hdemo_sk", "ss_addr_sk", "ss_store_sk", "ss_promo_sk", "ss_ticket_number", "ss_quantity",
        "ss_wholesale_cost", "ss_list_price", "ss_sales_price", "ss_ext_discount_amt", "ss_ext_sales_price",
        "ss_ext_wholesale_cost", "ss_ext_list_price", "ss_ext_tax", "ss_coupon_amt", "ss_net_paid",
        "ss_net_paid_inc_tax", "ss_net_profit"
    ]
}


path = "./data"
#path = "./temp"
out_dir = os.path.join(path, "conv")
os.makedirs(out_dir, exist_ok=True)   # conv 폴더 없으면 생성
csv_files = glob.glob(os.path.join(path, "*.csv"))

# 방법 2: os.listdir 사용 (필터링)

dat_files = [
    os.path.join(path, f)
    for f in os.listdir(path)
    if f.lower().endswith(".dat")
]

fix_none = lambda x: None if (x is None or x == '') else ("None" if x == "None" else x)

for csv_file in dat_files:
    base = os.path.basename(csv_file)              # person_info.csv
    out_file = os.path.join(out_dir, base)         # ./temp/conv/person_info.csv

    out_file = csv_file
    name, _ = os.path.splitext(out_file)
    table = os.path.basename(name)             # person_info
    print(table, out_file)
    read_kwargs = dict(sep="|", header=None, chunksize=500000, names=table_columns[table],
        engine="python", quotechar='"',
        doublequote=True,
        escapechar='\\',
        quoting=csv.QUOTE_MINIMAL,
        keep_default_na=False,
        na_filter=False,
        converters={col: fix_none for col in table_columns[table]}  # 모든 컬럼 적용
    )

    chunker = pd.read_csv(out_file, **read_kwargs)
    total_rows = 0
    for i, chunk in enumerate(chunker, start=1):
        rows = len(chunk)
        total_rows += rows
        try:
            chunk.to_sql(schema='tpcds', name=table, con=engine, index=False, if_exists='append')
        except Exception as e:
            if not os.path.isdir("./error/"):
               os.mkdir("./error/")
            chunk.to_csv(f"./error/1.csv")
            with open(f'./error/error.log', 'w', encoding='utf-8') as f:
                f.write(str(e))
                f.close()
            raise NotImplementedError
        print(f"[{i:04d}] chunk rows={rows:,}  processed={total_rows:,}") 
```