+++
date = '2025-09-02T11:52:47+09:00'
weight = 9
title = 'pgvector_build'
tags = ["Vector", "Database", "Embedding", "ANN", "Similarity Search", "postgresql"]
categories = ["Vector database","Study", "development"]
+++

# **pgvector build 하기**
## **기본 정보**
 - https://github.com/pgvector/pgvector
 - postgresql 이 먼저 running 한 뒤에 설치하여야한다.

### **postgresql 준비**
 - postgres 13이상 버전 준비
```sh
wget https://ftp.postgresql.org/pub/source/v17.6/postgresql-17.6.tar.gz
tar -vxzf postgresql-17.6.tar.gz
```
 - path 설정 및 build
```sh
cd postgresql-17.6.tar.gz
./configure --prefix=/home/{user}/data/postgresql-17.6 --without-readline
sudo make -j
sudo make install
```
 - postgresql 실행
```sh
# Create and start the DB
/home/{user}/data/postgresql-17.6/bin/pg_ctl -D ~/vector_test initdb

# change port_numver from 5432 to 5557
vim ~/vector_test/postgresql.conf

/home/{user}/data/postgresql-17.6/bin/pg_ctl -D ~/vector_test start -l vec_test_logfile

# create database
/home/{user}/data/postgresql-17.6/bin/createdb vector_test -p 5557

# verify connection
/home/{user}/data/postgresql-17.6/bin/psql vector_test -p 5557
```

### **pgvector 빌드**
 - pg vector (공식 README.md 는 v0.8.0 을 사용하는것을 권장하고 있으니 참조)
```sh
git clone https://github.com/pgvector/pgvector.git
cd pgvector
export PG_CONFIG=/home/{user}/data/postgresql-17.6/bin/pg_config
make
sudo --preserve-env=PG_CONFIG make install

# verify that extension is installed
/home/{user}/data/postgresql-17.6/bin/psql vector_test -p 5557
CREATE EXTENSION vector;
```

### **SIFT1M Dataset**
 - 공식사이트 : http://corpus-texmex.irisa.fr/
```sh
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -vzxf sift.tar.gz
```
 - table 생성
 ```sql
 CREATE TABLE sift1m (
  id  bigint PRIMARY KEY,
  vec vector(128)
);
 ```
  - load by python
```python
# save as fvecs_copy_stream.py
import numpy as np
from psycopg import connect

def read_fvecs_memmap(path):
    # 메모리 절약을 위해 memmap 사용
    a = np.memmap(path, dtype=np.int32, mode='r')
    d = int(a[0])
    n = a.size // (d + 1)
    a = a.reshape(n, d + 1)
    X = a[:, 1:].astype(np.float32, copy=False)  # view가 안 되면 astype with copy=False
    return X, d

def normalize_rows(X, eps=1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < eps] = 1.0
    X[:] = X / n
    return X

def stream_copy_fvecs(dsn, table, fvecs_path, normalize=False, batch=10000):
    X, d = read_fvecs_memmap(fvecs_path)
    assert d > 0

    if normalize:
        normalize_rows(X)

    with connect(dsn) as conn:
        copy_sql = f"COPY {table} (id, vec) FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t')"
        with conn.cursor() as cur, cur.copy(copy_sql) as cp:
            rid = 0
            N = X.shape[0]
            while rid < N:
                end = min(rid + batch, N)
                for i in range(rid, end):
                    v = X[i]
                    line = f"{i}\t[{','.join(f'{x:.6f}' for x in v)}]\n"
                    cp.write(line.encode('utf-8'))
                rid = end
        conn.commit()

if __name__ == "__main__":
    DSN = "host=/tmp dbname=vector_test port=5557"
    stream_copy_fvecs(DSN, "sift1m", "sift_base.fvecs", normalize=True, batch=20000)
```
```sh
cd sift
## above python script
pip install "psycopg[binary]>=3.1.18" 
python3 load_vec.py 
```
 - load 확인
```sql
vector_test=# select count(*) from sift1m;
  count
---------
 1000000
(1 row)
```
### **index build**
 - 
 ```SQL
CREATE INDEX sift_hnsw ON sift_base USING hnsw (vec vector_cosine_ops) WITH (m = 16, ef_construction = 200);
 ```