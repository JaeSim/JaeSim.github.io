+++
date = '2025-11-19T11:47:44+09:00'
title = '모바일 원격 와우하기'
tags = ["wow", "world of warcraft", "와우", "moonlight", "consoleport"]
categories = ["development", "life", "game"]
+++

# **핸드폰으로 원격 와우하기**
 - 요약 : 3가지 기술 & 장비의 조합 [ConsolePort add-on, Moonlight streaming, 모바일게임패드]
 - 궁극적인 goal은 "핸드폰에서 와우를 하기".  ("닌텐도 게임기에서 와우를 한다" 도 가능)

## **ConsolePort add-on**

- PC에서 와우를 할때 게임패드 (닌텐도 스위치 패드, 플스 패드) 등으로 하기 위한 add-on <br>
 ref : https://www.inven.co.kr/board/wow/1896/46534

- PC에 패드를 연결하고, ConsolePort 애드온을 사용해서 키를 맵핑하면, 콘솔게임용 패드로 와우 플레이가능
- 패드류 device를 인식하여 조작편의성을 높혀주는 addon

## **Moonlight Streaming**
 - PC에 있는 화면을 remote device에 원격 스트리밍 기술

 - geforce shield 에서 지원했었는데 <u>**더이상 지원하지 않고**</u>, Moonlight 라는 오픈소스 기술로 대체됨.
 - 핸드폰에서 moonlight app 설치가 필요하고, PC에서도 Broadcasting 설정이 필요.
 - 설정방법 :  https://www.postype.com/@hyerinxiv/post/18201314


 - 스트리밍으로 하기 때문에 반응성 이슈에 대한 우려가 있을 수 있으나 <br>
   (같은 집 wifi 내라면) 매우 빠르게 반응하며 게임 플레이에 지장이 없음
 - 이동시에는 시도해보지 않았지만 데이터 요금 및 품질 저하 이슈가 있을 것으로 보임

## **모바일 게임패드**
 - 싼것은 2만원대부터 비싼것은 10만원이 넘고, 보통 2~4만원대 가격 형성
 - gamesir X5 lite 사용 [중국 직구로 배송이 매우 느림]
 - 싼것으로 플레이해본 다음 맞는 다음 패드 구매 추천.
 - 당근에 중고 패드가 많음. 게임 플레이 특성상 한두번 해보고 안맞으면 창고에 썩히는 경우가 많은것 같음

## **종합**

1) 핸드폰에 모바일 게임패드를 연결
2) moonlight 로 pc 화면 전송
3) 와우 실행 -> consoleport 애드온으로 패드 조작


## **Appendeix**
### **느낀점1**
닌텐도 스위치가 custom firmware가 깔려있다면, moonlight app을 설치할 수 있어서, 바로 닌텐도 게임기에서 PC화면을 받을 수 있고, <br>
결과적으로 2)3) 이 되므로 추가 디바이스 구매 없이 바로 적용 가능


### **느낀점2**
 - 조작이 익숙해져야해서, 혼자하는 퀘스트 밀기, 구렁밀기, 공찾[조금 숙련] 정도는 무난히 플레이 가능
