+++
date = '2025-07-16T10:01:12+09:00'
title = 'Adaptive Watermark in Stream Processing'
weight = 7
tags = ["Stream Processing", "watermark", "adaptive", "Database"]
categories = ["Adaptive Watermark", "Stream Processing", "Study"]
+++

# **Adaptive watermark 관련 리서치**

## **Stream Processing**

## **Survey paper**
### **A survey on the evolution of stream processing systems(2023)**
#### **Summary**
 - VLDB23 저널, 35 pages
 - https://arxiv.org/abs/2008.00842
 - Stream Processing System(Solution)들을 1세대(92 ~ 04), 2세대(10 ~ 17), 3세대(18 ~)로 구분하여 설명하고, 어떤 문제점들을 해결하기 위해서 발전해온지 망라
   - 1세대 -> 2세대는 `MapReduce`와 Cloud Computing의 대중화로 인해 세대를 구분함
   - 2세대 -> 3세대는 serverless App, Cloud Server, Edge Computing 이나 Hardeware specific으로 변화하는 것으로 구분함 <br>
   구태여 3세대를 나누긴 했는데, 구체적으로 왜그런지 명확하게 언급되어 있진 않다.
   - 1세대 특징: 
     1) Complex Event Processing :CEP 처리에 중점
     2) scale-up arch 기반
     3) ordered event streams 처리방식
   - 2세대 특징:
     1) distributed 환경에서 data-parallel processing engine
     2) support MapReduce style UDF(User-Defined Function)
     3) scale-out for massive out-of-order stream (toward fault-tolerant)
   - 3세대 특징
     1) integrate data-streaming to (serverless App, Cloud Server, Edge Computing or Hardeware specific) <br>
    SQL기능을 추가하는 방식도 존재
 - Stream processing solution들이 채택한 기법들을 나열 및 open problem 기술
    - out-of-order 처리 기법, State management, Fault tolerance, load management, elasticity, reconfiguration 처리 기법 

## **out-of-order(disorder)**
 - **A survey on the evolution of stream processing systems(2023)** [[이동]](#a-survey-on-the-evolution-of-stream-processing-systems2023) 에 언급된 내용을 base로 작성
### **Basic**
 - stream이 순차적으로 도착하지 않고, 순서가 뒤바뀌어 적재된 상황
 - disorder 와 out-of-order 를 동의어로 기술
 - 발생원인
    - 주로 network issue로 인해 발생 
    - 그이외에는 특정 operator들이 disoder를 발생시킨다.
      - Join orperator : 매칭 순서대로 출력하기때문에 뒤섞임
      - windowing : 정렬기준이 아니라 속성 기준으로 windowing하면 순서가 뒤바뀜
      - data prioritization : 정렬 기준이 아니라 다른 속성으로 prioritization 하면 순서가 변경
      - union operator : union 으로 합치면 무작위로 뒤석인 스트림이 생성
 - stream processing system은 disorder를 해결하기 위해 ***detect*** 와 ***processing progress를 측정(mesasure)*** 할 수 있어야 한다.
   - 여기에서 _progress_ 는 얼마나 stream이 처리가 되었는지를 나타냄. 
   - 통상적으로, 특정 값 A 를 기준으로 작은 값부터 처리하도록 해서, A의 최소값이 자동적으로 진행의 척도(measure of progress) 가 되도록 한다.


### **Two main architectures for disorder : in-order processing , out-of-order processing**
 - 최근에는 out-of-order processing이 대세로 보임
 - in-order, out-of-order 말고도 Revision Processing 도 연구중

#### **in-order processing**
 - 3가지 전략중 하나를 택함
 1) [**늦는거 폐기**] 입력 스트림이 정렬되어 있다고 가정하고, late data를 폐기(discard)한다
 2) [**최대한 정렬**] 입력 스트림을 버퍼에 저장하고, lateness bound(허용 지연: 얼마나 늦은 record까지 허용할지) 내에 재정렬 후, 정렬된 record 처리후 buffer clear
 3) [**정렬 안한채 적재**] 재정렬 없이 stream을 허용한다. 이는 이후 연산자(하위연산자라고 표현) 사용할때, 각 연산자가, 독립적으로 허용 지연(lateness bound)를 고려해서 처리하도록 해야한다
 - 결과적으로 stream이 정렬되어 들어온다고 가정하고, 이를 기반으로 progress를 추정한다.

#### **out-of-order processing**
 - operator or global authority (전역 제어자) 가 **progress tracking mechanism** 을 사용해서, progess information을 생성하고, 이를 dataflow graph에 전파한다. <br>
   progress information은 아직 처리되지 않은 가장 오래된 레코드를 나타내고, 허용지연도 설정한다.
   - **progress tracking mechanism** : `slack`, `heartbeat`, `watermark`, `pointstamps` <br>
    아래에서 설명 [[이동]](#mechanisms-for-managing-disorder)
 - 허용지연을 넘지 않은 disorder는 지연없이 순서대로 처리가 가능

### **Effects of disorder**
 - out-or-order 데이터를 in-order/out-of-order sytstem에 전달될 경우
   - in-order system에서는 재정렬을 위해서 processing overhead, memory space overhead, latency를 초래
   - out-of-order system에서는 progress를 추적하면서, 입력스트림이 disorder라도 lateness bound 안에 있으면 처리하고, bound보다 오래된 데이터는 remove한다
 - order-sensitive 연산자들이 영향을 받음.
   - 입력 데이터의 일부만 계산하여 결과가 정확하지 않음

### **Mechanisms for managing disorder**
 - *slack and heartbeat는 최신 솔루션에서는 잘 채택되지 않음*
 
<img src="/images/out-of-order-tracking-methods.png" alt="out-of-order-tacking-methods" style="width:80%;" />


#### **Slack**
 - 고정된 시간 또는 개수만큼 기다리는 simple mechanism
 - 위 이미지에서 `t=4`가 수신되면 다음 한개는 기다림 (slack=1 로 설정되어 있어서)
 - `t=5`의 경우 무시 (slack=1 을 초과해서)

#### **Heartbeat**
 - progress information을 담은 외부 입력 신호(external signal).  `h=2` 라면 2까지는 올것들이 다 전송되었다 라는 의미
 - input source 로부터 명시적으로 전송되거나 (iot 센터가 주기적으로 heartbeat를 보냄) 
 - 시스템환경 정보를 통해서 추론해서 끼워 넣을 수 있음
 - heartbeat 시그널이 오면 타임스탬프 이하의 레코드들을 연산자에게 전달. 위 그림에서는 `h=2`, `h=4` 가 왔을때 어떻게 처리되는지 나타낸 것

#### **Low-watermark and punctuation**
 - **punctuation**
   - 데이터 레코드처럼 삽입되는 명시적 메타 데이터
   - 더 이상 특정 범위의 데이터는 도착하지 않음을 선언하는 메타데이터 신호이며, 진행 추적·윈도우 종료·상태 정리 등 **다양한 연산을 트리거하는 데 사용**  된다.
 - **watermark**
    - 시스템이 추론해서 자동생성하는 암시적 메타데이터
    - 시스템에서 지금까지 도착했을 것으로 간주되는 oldest timesteamp 기준으로 생성
    - watermark 보다 같거나 큰 값들만 disoder를 처리. 작으면 무시
    - 워터마크와 tumbling window 사이즈와 맞물려서 window가 언제 닫히고 연산(처리)되어야하는지 가 결정된다.
 - watermark가 punctuation 의 하위셋 이라고 추상적으로 이해해도 될것 같다.

#### **pointstamps**
<img src="/images/out-of-order-tracking-methods-2.png" alt="pointstamps"  />

 - watermark보다 좀더 정밀한 추적을 위해서 제안된 기법
 - 연산자 OP1, OP2, OP3 이 있다고 가정하고 (Source, filter, aggr, 등)
   - 각 연산자가 어디까지 처리할 수 있는지를 나열
 - pointstamp(t:timestamps, l:opertor location) 를 부착하는데 location은 edge나 node 둘다 될 수 있음.
 - frontier와 함께 사용되고, frotier를 통해서 어디까지 처리를 할수 있는지 나타냄.

### **cyclic query**
 - join이나 union 같은 binary operator(이항연산자)가 있어서, 하나의 출력이 다시 입력이 되면, 처리를 기다려야하는 상황이 발생
 1) 이를 해결하기위해서 dedicated operator 를 제안 (Chandramouli et al.)
    1) 지나간 event들의 timestamp 기반으로 speculative punctuation(추론적)를 삽입(전용 operator)
    2) 되돌아 왔을대, 잘 도착했는지 관측하고, 유효하면 valid punctuation으로 변환(speculative punctuation을 valid한것으로 마킹한다고 이해)하여 전달
 2) Naiad 에도 별도 솔루션을 제안했는데
    - 이벤트 핸들러는 현재 처리중인 이벤트보다 큰 timestamp를 가진것만 처리하다는 제약을 두어서, 전체 pending event들의 cycle로부터 partial order 문제로 변환. 이로인해 2)가 가능해짐
    1) loop가 돌면 카운터를 증가되는 특수 연산자를 넣고,
    2) earliest logical time을 계산하여 notification을 전달

### **Revision Processing**
 - 이전 ouput을 모종의 이유로 (late arriaval, updated, retract(취소)) 로 정정(revise)해야하는 경우
 - `Store and Recivse`, `Replay and Revise`, `Partition and Consolidate` 방법등이 있다.

## **Watermark**
 - disorder 를 해결하기 위한 기법
## **Adaptive watermark**
 - 동적으로 watermark를 설정하는 관련 페이퍼 모음
### **Adaptive Watermarks: A Concept Drift-based Approach (2019)**
 - EDBT short paper(4 pages)  [분포기반]
 - https://openproceedings.org/2019/conf/edbt/EDBT19_paper_211.pdf
 - Adaptive Window(ADWIN) 알고리즘을 통해서 데이터 도착의 변화 및 지연의 변화를 감지해서, window를 더 빨리(늦게) 트리거 되도록 변경시킴. 이를 통해 latency 를 줄이고, data drop rate를 낮춤
  
### **Adaptive watermark generation mechanism based on time series prediction for stream processing(2020)**
 - Springer 
 - 시계열 예측 모델을 통해서  [통계기반]