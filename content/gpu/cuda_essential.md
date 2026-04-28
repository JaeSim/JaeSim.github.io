+++
date = '2026-04-17T12:21:36+09:00'
weight = 1
title = '1. CUDA Essential'
tags = ["GPU", "NVIDIA", "ML", "essential", "CUDA"]
categories = ["GPU", "NVIDA", "CUDA"]
+++

# **1. CUDA Essential**

## **CUDA 란 무엇인가**


### **CUDA programing**
Reference : https://docs.nvidia.com/cuda/cuda-programming-guide/index.html

**CUDA (Computed Unified Device Architecture)** 는  개발한 GPU 개발 툴 이다.
NVIDIA에서 GPU를 활용한 병렬 프로그래밍을 가능하게 만드는 소프트웨어만드는 C++기반의 언어 (소프트웨어 플랫폼)


Nvidia는 GPU를 ***device*** 라고 지칭하고, GPU에서 동작하는 application code를 ***device code*** 라고 명하며, <br>
GPU에서 실행하는 function을 ***kernel*** 이라 명한다. 


GPU는 SM의 모음으로 보아도 된다.

SM (Streaming Multiprocessor)  : GPCs (Graphics Processing Cluster) 들의 모음

각 SM은 local register 파일과, unified cache (L1 cache 와 shared memory), 컴퓨팅하는 functional unit들 이 포함되어있다.

<img src="/images/gpu/gpu-cpu-system-diagram.png" alt="gpu_sm_diagram" style="width:90%; background-color:white;" />

kernel은 여러 thead를 포함할 수 있고, 이 thread 의 묶음을 block, 그리고 이 thread block의 묵음이 grid이다.<br>
grid 안의 thread block은 같은 사이즈와 차원을 가진다. <br>
커널이 실행될때, execution configuration을 같이 쓰고, <br>
이것에는 grid나 thread block dimension, 클러스터사이즈나 SM 설정같은 옵션등이 포함될 수 있다.

thread block과 grid는 차원을 가지는데, 이것이 작업단위와 데이터 맵핑에 이점을 준다고 소개한다.

각 thread는 내장 변수가 있는데, 자신이 어느 block에 속해있는지, 어느 grid에 속해있는지를 저장하고, kernel이 이것을 참조 할 수 있다

모든 thread는 하나의 SM 내에서 동작한다 <br>
thread 들은 block내 thead와 동기화가 가능하다.<br>
thread는 on-chip shared memory에 접근할 수 있고, 다른 thread block내 thread와 정보 교환 가능하다


<div style="display: flex; gap: 10px; text-align: center;">
  <div style="flex: 1;">
    <img src="/images/gpu/thread-block-scheduling.png" alt="thread_block_assign" style="width:90%; background-color:white;" />
    <p><strong>thread assign</strong></p>
  </div>
  <div style="flex: 1;">
    <img src="/images/gpu/thread-block-scheduling-with-clusters.png" alt="thread_block_assign_gpc" style="width:90%; background-color:white;" />
    <p><strong>thread assign with GPC</strong></p>
  </div>
</div>


thread block간의 의존성은 보장할수 없고, output을 가져다가 쓸수도 없다.

grid안의 thread block은 cluster로 그룹핑이 가능하다. <br>
GPC (global platform cooperative ) 기술이라 명명.

GPC로 묶여있으면, block간 cooperative group의 interface를 통해서 동기화가 가능해지며, <br>
 cluster 내부 thread들은 클러스터내부 블록들의 distributed shared memory에 접근이 가능하다.




### **Single-Instruction Multiple-Threads (SIMT)**

thread block안에서 32개의 thread block을 ***warp*** 라고 한다

워프내의 모든 thread는 동일한 커널 코드를 실행하지만, 코드내 다른 branch(분기)를 따라 갈수 있음

<img src="/images/gpu/active-warp-lanes.png" alt="active-warp-lanes" style="width:90%; background-color:white;" />

일부는 마스킹 되어서 처리되는데 이를  warp divergence  라고 지칭한다고 한다.

되도록 한가지 분기를 잘 타고갈때,(한가지 흐름으로 코드가 진행될때), 활용도가 최상이 된다는 점을 유념하자

warp에 있는 thread는 lock step(동시에라고 해석) 으로 수행된다. 독립적으로 수행도 가능하긴하다<br>
https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-independent-thread-scheduling
<br>
3.2.2.1.1. Independent Thread Scheduling 를 참조<br>

nvidia는 warp가 real HW에서 어떻게 맵핑되는 것을 활용하는것은 권장하지 않는다


warp를 위해서 thread block내의 thread수가 32배수가 되는것이 가장 좋다고 언급한다 <br>
32배수가 아니면 warp단위로 동작하는데, 빈 곳이 있을 수 있다.


SIMD(Single Instruction Multiple Data) 와 SIMT는 다른 것으로,
SIMD는 단일 제어 흐름만 따르지만, SIMT는 분기가 가능하다. (그러나 분기가 될수록 성능저하가 발생하니, 일단 가능하다는 옵션이 있다고만 이해하자)

### **GPU Memory**

GPU도 on-chip 메모리를 가지고 있다. <br>
Nvidia를 gpu에 부착된 DRAM을 global memory 라고 지칭한다. GPU의 모든 SM이 접근이 가능하다.

CPU에 부착된 DRAM을 host memory , system memory 라고 부른다.

#### **DRAM memory with GPU & CPU**
- ***CPU와 GPU는 통합된 단일 virtual memory space로 관리된다***

시스템 입장에서의 여러개의 GPU가 있을때, 어느 block이 어디에 있는지를 구분해낼수 있다.

CUDA API를 통해서 CPU GPU 간의 메모리 복사가 가능하다. GPU 간의 복사가 가능하다.

#### **on-chip memory in GPU**
각 GPU에 on-chip memory가 있고, SM들은 자체 레지스터 파일 및 공유 메모리를 가지고 있음 <br>
이 레지스터 파일과 shared memory 이 SM내 thread는 빠르게 접근 가능하나 다른 SM에서 실행하는 thread는 접근 불가

- 레지스터 파일에는 컴파일러가 할당하는 스레드 지역변수가 저장됨<br>
레지스터 할당은 thread 단위로 가능
- 공유 메모리는 thread block과 클러스터 내 thread들은 접근 가능하고, 데이터 교환에 사용 가능 <br>
shared memory는 thread block 단위로만 할당 가능

##### **cache in on-chip memory**
GPU에는 l1,l2 캐시도 가지고 있다

- L2 cache는 GPU내 모든 SM이 공유하는 cache
- L1 cache는 각 SM이 일부분씩 가지고 있다
- SM별로 분리된 constant cache가 있고 kernel이 contant로 선언한 것들이 life타임동안 올라간다.

compiler 가 커널 파라미터를 상수로 올려둘 수 있다.

### **CUDA Toolkit**

GPU 컴퓨팅을 활용하는 library 와 header, wrting, building, software 분석 의 모음

The CUDA Toolkit is a set of libraries, headers, and tools for writing, building, and analyzing software which utilizes GPU computing. The CUDA Toolkit is a separate software product from the NVIDIA driver

### **CUDA Runtime**

CUDA tollkit에서 제공하는 library 중 하나로써 <br>
<u>**메모리 할당, 데이터 복사(gpu-gpu, gpu-cpu), kernel 실행**</u> 하기 위한 일부 API와 language extension 을 제공한다

#### **CUDA Runtime API**
CUDA runtime의 API 구성요소를 보통 CUDA runtime API 라고 칭한다.

이것은 NVDIA Driver에서 노출된 CUDA Driver API위에 구현된것으로 <br>
CUDA Driver API는 lower-level API이다.

// 아래 링크에서 추가 공부 필요
https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api


### **Parallel Thread Execution (PTX)**

Nvidia GPU의 high-level assembly language 가 Parallel Thread Execution (PTX) 으로 <br>
virtual inststruction set artitechture (ISA) 이다

고수준 assembliy 언어로 GPU HW의 physical ISA 위에 추상화를 제공

PTX코드를 중간표현단계(intermediate representation)으로 만든 뒤, offline 이나 JIT(Just-in-Time) 컴파일로 실행가능한 바이너리 GPU 코드를 만든다.

NVCC (Nvidia CUDA Compiler) 만이 아니라 다른 것들도 PTX를 생성하여 GPU 프로그래밍 가능


### **Cubin & Fatbin**

CUDA application 이나 library는 일번적으로 C++로 작성하고, <br>
PTX파일로 컴파일 되고, PTX는 물리적 GPU에서 쓰는 실제 binary 파일로 컴파일된다. <br>
마지막 파일을 CUDA binary 나 cubin으로 부른다
SM 버전마다 맞는 Cubin 버전이 있다.


GPU 실행 파일과 라이브러리 바이너리에는 CPU와 GPU 코드가 혼재되어있는데 <br>
GPU 코드는 fatbin 이라는 곳에 저장된다

fatbin에는 여러 cubin 파일들과 PTX 파일들이 저장될 수 있다. <br>
GPU코드가 멀티플 GPU 아키텍쳐를 포함하여 함께 빌드될 수 있는데, 어플리케이션이 실행하면 fatbin에서 <br>
가장 적합한 binray가 실행됨

### **Compatibility & Just-in-Time compilation**
- 호환성 : major.minor 버전이 있다면 major버전이 차이나면 호환이 안된다 
- minor는 하위버전만 빌드된 cubin이 상위버전 GPU에서 동작한다
- 예   8.6 cubin 은 8.6 ~ GPU에서 동작한다. (그러나 9.0~부터는 동작 X)

PTX 코드는 어플리케이션 실행시 JIT로 컴파일 될 수 있다.

실행시간에 PTX 코드는 device driver에 의해서 바이너리 코드로 컴파일

이때 생성된 binary code copy를 자동으로 compute cache에 된다.


## **CUDA Compile**
### **NVCC NVidia CUDA Compier**

Reference : https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html

***non-CUDA phase*** 와 ***CUDA phase*** 로 나뉜다


### **CUDA Programming**

host에서 호출되면서 GPU에서 실행되는 것은 kernel 이다.

- ``__global__`` 이라고 선언해서 gpu 코드라고 컴파일러에게 알려줘야한다
- ``<<< >>>`` triple chevron notation으로 부르는데 parameter 를 지정한다

```c++
 __global__ void vecAdd(float* A, float* B, float* C)
 {

 }

int main()
{
    ...
    // Kernel invocation
    vecAdd<<<1, 256>>>(A, B, C);
    ...
    dim3 grid(16,16);
    dim3 block(8,8);
    MatAdd<<<grid, block>>>(A, B, C);
}
```
- triple schevron 에서 첫번쨰 두번쨰는 grid dimension과 thread block의 dimension을 나타낸다.

- thread blcok은 최대 1024 thread 를 포함 가능하다
- block의 크기, thread 크기 등을 고려해서 넘치지 않게 잘 코드 짜야한다

### **Unified Memory**


- ``cudaMallocManagedAPI`` 나 ``__managed__`` 를 통해서 선언한다
- ``cudaFree`` 를 통해서 해제한다.
- address translation service나 heterogeneous memory management 를 사용하면 자동으로 해주기 때문에, ``cudaMallocManagedAPI`` 나 ``__managed__`` 가 필요 없다

- ``cudaMalloc`` 으로 명시적으로 관리가 가능하다 (해제는 동일하게 ``cudaFree``) <br>
그러나 코드가 더 verbose 해진다

- ``cudaMemcpy`` 를 통해서 CPU->GPU 로 buffer 를 copy한다. <br>
  마지막 파라미터가 ``cudaMemcpyKind_t`` 인데, 다음과 같은 값을 가질 수 있다
  - ``cudaMemcpyHostToDevice`` : CPU -> GPU
  - ``cudaMemcpyDeviceToHost `` : GPU -> CPU
  - ``cudaMemcpyDeviceToDevice `` : GPU -> GPU
- ``cudaMemcpyAPI`` 은 synchronous 로 되는점을 참고해자
- ``cudaMallocHost`` 을 통해서 host에 page-locked 메모리를 할당 할 수 있다. <br>
보통 asynchronous 메모리 전송에 사용한다. CPU-> GPU 할때 이걸 사용하자


### **Synchronizing CPU and GPU**
kernel 실행은 비동기로 진행되니 ``cudaDeviceSynchronize`` 를 이용하자 <br>
gpu 일이 끝날떄까지 host thread가 block된다.

- Stream synchronization API와 CUDA even를 이용해서 asynchronouse Execution 하는 것이 가능하다


Thread block내의 Thread는 shared memory를 통해서 메모리 access 동기화가 가능하다<br>

- ```__syncthreads``` 내장함수(intrinsic function)을 통해서, thread block내 동기화가 가능하다. <br>
thread block 외부는 cooperative groups로 동기화 메커니즘을제공한다.


### **CUDA runtime initialization**

CUDA runtime은 ```CUDA context```를 각 device 에 생성한다.<br>
(main 함수라고 이해하였다)
CUDA context host(CPU)에 모든 thead에 공유되고, Context 생성에 필요한 device code(GPU code) 는 JIT 컴파일되고, 필요시에 device memory에 로드된다.

Context는 drvier API를 호출할 수 있다.

CUDA 12.0 ~ 부터는 ```cudaInitDevice``` , ```cudaSetDevice``` 를이용해서 context 초기화를 한다<br>
(12.0 이전버전에서는 ```cudaSetDevice``` 은 초기화하지 않음)

```cudaDeviceReset```로 Context를 날릴 수 있다


### **Error Checking in CUDA**
모든 CUDA API는 ```cudaError_t``` 형태의 값을 반환하는데, 오류가 없으면 ```cudaSuccess`` 이다.
기본값은 ```cudaSuccess``` 이고, 오류가 발생하면 덮어씌어진다.<br>
```cudaGetLastError``` 이나 ```cudaPeekLastError``` 로 마지막 에러를 확인하자

- triple chevron notation ( ```<<< >>>``` )  은 에러를 반환하지 안흔다.


### **CUDA_LOG_FILE**
CUDA_LOG_FILE 는 환경 변수이며, 다음과 같이 사용된다

```sh
$ nvcc errorLogIllustration.cu -o errlog
$ ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument

# or
$ env CUDA_LOG_FILE=cudaLog.txt ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
$ cat cudaLog.txt
[12:46:23.854][137216133754880][CUDA][E] One or more of block dimensions of (4096,1,1) exceeds corresponding maximum value of (1024,1024,64)
[12:46:23.854][137216133754880][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuLaunchKernel
```

### **keyword**
- `__global` kernel이 진입하는 지점(entry point)을 나타내기 위해서 사용
- `__device__`  GPU로 컴파일 되어야함을 설정
- `__host__` cpu로 컴파일되어야 한다고 설정
- `__constant__` 변수가 전역메모리 저장하도록 설정
- `__managed__` 변수가 통합(Unified) 저장하도록 설정
- `__shared__` 변수가 공유메모리(shared) 저장하도록 설정
- `__host__ __device__` 로하면 GPU, CPU 둘다 코드가 컴파일된다 <br>
지원되는 GPU를 `__CUDA_ARCH_` 로 확인 필요

### **Thread Block Cluster**

thread block이 같은 cluster에 묶이면 GPU Processing Cluster (GPC) 로 같이 스케쥴링된다.

최대 8개 thread block이 클러스터링 가능한데, GPU 타입에 따라서 or MIG로 쪼개면 달라질 수 있으니 `cudaOccupancyMaxPotentialClusterSizeAPI` 로 확인 필요.
- `cluster.sync()` 라느 cooperative group API로 동기화가 가능하다.
- `num_threads()`, `num_blocks()` 로 cluster 내 thread와 block 사이즈를 알 수 있다.
- 내부 순서는 `dim_threads()`, `dim_blocks()` 로 알아 낼수 있다

코드에서 thread block cluster를 사용하려면  `__cluster_dims__(X,Y,Z)` 을 하거나 `cudaLaunchKernelEx` 를 사용해야한다

- `__cluster_dims__(X,Y,Z)` 는 compile-time 속성을 사용해서, 컴파일시에 크기가 고정되어 나중에 클러스터 크기를 수정 못한다. 대신 고정되어 `<<< >>>` 로 사용가능
- `cudaLaunchKernelEx` 는 JIT 방식이다.