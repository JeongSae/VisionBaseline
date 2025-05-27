# VisionBaseline

---
작업 내용
---
## 2025 05
1. Hydra 적용
2. Training 종료 후 레포트 출력
   1. micro/macro-average ROC와 PR 커브를 계산
3. Eval 함수 작성
4. Training 과정 내 메모리 증가 이슈 해결 (gc, no grad)
5. wandb 연동, runs/train 및 runs/train/weights 생성 및 관리 추가
   1. Hydra 폴더 중복 생성 수정 -> Main() 내부로 이동

---
남은 작업 리스트
---
1. Weight intialization 방법 선택 (모델 레이어 내 공통 적용)
2. Dataset 선택 ( mnist, cifar100, flower, imagenet 등 다운로드 후 압축 풀도록 코드 수정 )
3. Device, gpu parallel 선택 ( single, multi, DDP )
4. 학습/검증 루프 분리 및 재사용성 개선
   1. 리팩토링
5.  LR 스케줄러 도입 + 기타 스케줄러
6.  타입 힌트 & Docstring 추가 (함수 별)
7.  메트릭 초기화/집계 로직 개선

---
구현 예정 함수 리스트
---
1. ImageNet, Cifar10, Cifar100 등 기본 데이터 셋 제공
2. Multi-Scale Crop function (Optional) 추가 -> IPYNB 구현 완료, 추후 Test 단계 적용 예정
3. Dense Evaluation function (Optional) 추가
4. VGG 전체 점진적 학습 / Transfer Learning (Optional) 추가
5. 데이터 증강 파라미터화 + 기본 데이터 증강 추가
6. ResNet 구현