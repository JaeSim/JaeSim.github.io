+++
date = '2025-09-04T10:11:10+09:00'
weight = 10
title = 'WSL을 이용한 C언어 개발환경 셋업'
tags = ["wsl", "c", "c++", "vscode"]
categories = ["development"]
+++


# **WSL을 이용한 C언어 개발환경 셋업 하기**
## **기본 정보**
### **WSL**
- 세상이 많이 변했다. window 위 에서 손쉽게 linux 환경을 구축하는 것 정도로 이해하면 될 것 같다<br>
WSL2부터는 hyper-v를 이용하는것으로 이해

#### **WSL install**
 - target PC: Window 11 home, LG gram. 
 - powershell에서 아래명령어 실행. 자동으로 wsl2가 설치되는것으로 이해
 ```sh
 wsl --install
 ```
 - 에러에 대한 해결. 재부팅후 wsl-in
```sh
필요한 기능이 설치되어 있지 않기 때문에 작업을 시작할 수 없습니다.
오류 코드: Wsl/InstallDistro/Service/RegisterDistro/CreateVm/HCS/HCS_E_SERVICE_NOT_AVAILABLE
```
 - powershell을 관리자 권한으로 실행후 아래 커맨드 실행
 ```sh
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
dism.exe /online /enable-feature /featurename:HypervisorPlatform /all /norestart
 ```

#### **vscode 와 통합**
 - vscde 에서 wsl extension 
 - wsl에 gcc 설치
 ```sh
sudo apt-get update 
sudo apt-get upgrade
sudo apt-get install gcc
gcc -version
 ```
 - wsl shell로 사용할 폴더로 입력해서 vscode 키기
 ```sh
 cd {your path}
 code .
 ```
 - c++ extension 설치
 - vscode 에서 Terminal -> Configure Default build Task 에서 g++을 설정  (task.json을 설정하는 것 : build시 참조할 json)
 - build : shift +B ,  run : f5, debug : 체크포인트 (빨간점) 생성후 f5
 - launch.json을 만들기 (실행시 참조할 json). Run, add configuration.. 하면 대충 뼈대가 만들어지는데. <br>
   이중 `"args": ["<", "input.txt"],` 부분이 input file을 프로그램 시작시 넣어줄수 있도록 커스터마이징 한 것
 ```json
 {
    "configurations": [
    {
        "name": "(gdb) Launch",
        "type": "cppdbg",
        "request": "launch",
        "program": "${fileDirname}/${fileBasenameNoExtension}",
        "args": ["<", "input.txt"],
        "stopAtEntry": false,
        "cwd": "${fileDirname}",
        "environment": [],
        "externalConsole": false,
        "MIMode": "gdb",
        "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
                "ignoreFailures": true
            },
            {
                "description": "Set Disassembly Flavor to Intel",
                "text": "-gdb-set disassembly-flavor intel",
                "ignoreFailures": true
            }
        ]
    }
    ]
}
 ```