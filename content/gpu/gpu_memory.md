+++
date = '2026-04-24T14:49:24+09:00'
title = '2. GPU Memory'
weight = 2
tags = ["GPU", "NVIDIA", "ML", "Memory", "CUDA"]
categories = ["GPU", "NVIDA", "CUDA"]
+++

# **2. GPU Memory**

Reference : https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html

| Memory Type | Scope  | Lifetime     | Location |
|-------------|--------|--------------|----------|
| Global      | Grid   | Application  | Device   |
| Constant    | Grid   | Application  | Device   |
| Shared      | Block  | Kernel       | SM       |
| Local       | Thread | Kernel       | Device   |
| Register    | Thread | Kernel       | SM       |


참고: <br>
application 은 `main()`으로 시작한다 <br>
cuda kernel은 `<<< >>>` 로 실행된다. <br>
cuda context는 gpu를 실행시키는 환경이라고 생각하자. <br>

Application<br>
 └─ CUDA Context<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ Kernels (여러 개 실행 가능)


## **Global Memory**
커널의 모든 thread가 접근 가능한 persistent 메모리 <br>
application이 끝날때나 `cudaDeviceReset` 로 초기화 가능하다

`cudaMalloc` 과 `cudaMallocManaged` 으로 선언이 가능하다. <br>
`cudaMemcpy` 로 CPU->GPU copy가 가능하다

`cudaFree`로 free 한다.

## **Shared Memory**
Thread block 엤는 thread 가 접근 가능한 메모리 <br>
SM 별로 위치해 있고, Unified cache인 L1 cache와 같은 physical resource를 사용한다 <br>

- *_공유 메모리를 사용하면 커널에서 사용가능한 L1 cache 가 줄어들수 있다._*
- kernel에서 공유 메모리 사용 안하면, L1 cache에 할당된다.
- `cudaFuncSetCacheConfig` 를 이용해서 할당한다.
- `cudaGetDeviceProperties` 를 사용해서, `cudaDeviceProp.sharedMemPerMultiprocessor` 와 `cudaDeviceProp.sharedMemPerBlock` 를 확인 할 수 있다.
race condition이 있을수 있다. `__syncthreads()`를 통해서 동기화

- `__shared__`  를 사용해서 static 하게 메모리를 할당 할 수 있다. kernel 시간동안 유지
- dynamic 하게 할당하려면, `functionName<<<grid, block, sharedMemoryBytes>>>()` 으로 triple chevron notation하고 <br>
`extern __shared__` 를 통해서 동적으로 kernel이 메모리를 할당하게 할 수 있다. <br>
 포인터 연산을 사용해서 수동분할해야한다고 한다. 

## **Registers**
SM에 위치하고 컴파일러에 의해서 관리되며, 커널 실행중에는 thread의 local storage로 사용된다. <br>
NVCC를 통해서 최대 regiter 갯수를 설정할수 있다 `-maxrregcount`

## **Local Memory**
NVCC가 관리하는 Register과 비슷하게, thread의 local storage이나, <br>
**physical location은 global memory space에 있다**. <br>
local은 로지컬 local을 의미한다.

컴파일러가 자동으로 매치하는데
- constant quantity(상수 값)로 인덱싱이 되어있는지 확인 할 수 없는 Array
- register 공간을 너무 많이 차지하는 큰 구조(or 배열)
- any variable , 커널이 사용가능한 register를 초과하면 저장한다 (=register spilling)

로컬 메모리는 global memory와 동일한 latency와 bandwidth를 가진다. <br>
`Coalesced Global Memory Access`를 따르지만,
연속적인 Thread id의 연속적인 32bit word로 배치되기 대문에, warp 내의 접근은 상대주소로 병합된다. (= 따라서 효율적으로 수행된다)

## **Constant Memory**
grid scope 이며 어플리케이션 수명동안 접근 가능<br>
kernel에서도 접근이 가능한데 *read-only* 다

- `__constant__` 로 host측에서 지정하고 초기화해야한다
- 고정메모리 공간에 존재
- CUDA context 수명동안 유지
- device 별로 고유하게 가지고 있음
- grid 내 thread들을 다음것으로 접근 가능 `cudaGetSymbolAddress()`/ `cudaGetSymbolSize()`/ `cudaMemcpyToSymbol()`/ `cudaMemcpyFromSymbol()`
- 읽기 전요의 소량 쓰길 권장. 일반적으로 64KB

## **Caches**

GPU는 ***L1 과 L2 cache를 가지는 multi-level cache 를 가지고 있다.***

L2 는 ***_***모든 SM에 공유*** 되며 `l2CacheSize` 로 조회 가능.

L1 은 SM 에 physical리 located 되며, shared 메모리랑 같이 사용. <br>
L1 cache는 커널에서 shared memory로 사용하지 않으면 모두 L1 cache가 된다.

cudaGetDeviceProperties

## **Distributed Shared Memory**

Cooperative Group으로 묶인 cluster 내부에서는 thread들이 partitioned된(분할된) shared memory를 접근이 가능하다.<br>
이것을 **Distributed Shared Memory** 라고 지칭한다 . 그리고 그 주소를 distributed shared memory address space 라고 한다

distributed shared memory address space를 통해서 thread는 다른 thread block에 있더라도 접근이 가능해진다.

참조하려면 thread block들이 올라와 있어야한다. (=실행중에 있어야한다)<br>
`cluster.sync()` 로 확인 가능


## **Memory Performance**


32 byte 단위 memory transaction으로 처리되기 때문에 4byte word를 사용해도 32byte로 읽어간다.
따라서 잘 병합을 고려해서 코딩해라.
<img src="/images/gpu/perfect_coalescing_32byte_segments.png" alt="coalescing_example" style="width:90%; background-color:white;" />

다음은 간단한 예시
```c++
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
```

byte transfer에 사용되는 byte수를 최대화 하는것이여서, 위는 예시일 뿐


### **bank confilct**
메모리는 32bit의 대역포를 가지는 bank에 저장된다 (각각의 slot이라고 보면 될듯)
그렇기에 연속적인 32 word가 접근하면 32개의 bank가 접근하는데, 만약 conflict가 발생하면
순차적으로 진행된다.
따라서 성능저하를 피하기위해서, 잘 고려해서 코딩해야하며 다음과 같은 예외가 있다

- 여러 thread가 동시에 read할 경우 : broadcast됨
- 여러 thread가 동시에 write할 경우 : 하나만 write됨 (어떤건지 보장 안함)


https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-bank-conflicts 에 선언방식의 차이만으로도 bank conflict 를 해결하는 예제가 있다.

### **Atomics**
전체 grid level에서 전체 thread를 동기화 하는 방법은 현존하지 않음<br>
atomic function을 사용해서 lock하고, read-modify-write 하는 방식이 가능하긴 함

`cuda::std::atomic` 와 `cuda::std::atomic_ref` 인데 c++ 표준의 라이브러리와 동일함. <br>
thread scope 를 정할수 있는 atmoic을 function   `cuda::atomic` 와 `cuda::atomic_ref`  도 제공


### **Kernel Occupancy**
hardware resource와 설정에 따라서, SM이 사용가능한 resource가 달라진다. <br>
scheduler는 SM에게 여유가 없을때까지 할당하고, 현재 작업이 완료되면 thread block을 할당하는 작업을 지속한다.

결과적으로, SM에서 점유(할당)된 커널이 많도록 유지하면 성능은 올라간다.

block당 register 수,  SM이 사용가능한 용량,  block내 thread 수 등을 조합해서 점유율 을 최대로 올리는 optimizing이 가능하다

참조: 
https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#kernel-launch-and-occupancy
