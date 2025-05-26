# VisionBaseline
---
코드 수정 리스트
---
1. Weight intialization 방법 선택 (모델 레이어 내 공통 적용)
2. Dataset 선택 ( mnist, cifar100, flower, imagenet 등 다운로드 후 압축 풀도록 코드 수정 )
3. Device, gpu parallel 선택 ( single, multi, DDP )
4. report 출력 ( 그리드 형태로 예측, 예측 이미지, confusion matrix, roc-auc score, pr curve )
5. Evaluation 코드 추가
6. Validation 단계는 torch.no_grad() 블록으로 감싸기
7. DataLoader 설정 (drop_last, shuffle)
8. 학습/검증 루프 분리 및 재사용성 개선
9. LR 스케줄러 도입 + 기타 스케줄러
10. 로깅 및 시각화 (텐서보드, wandb)
11. Config 관리 개선 (Hydra)
12. 타입 힌트 & Docstring 추가 (함수 별)
13. 메트릭 초기화/집계 로직 개선
14. ultralytics 개념처럼, 모델 세팅과 관련된 내용들 제안
15. 실험 관리 및 학습 완료 내용 시각화
    1.  runs -> metrics, weight 등

---
추가 기능 함수 리스트
---
1. ImageNet, Cifar10, Cifar100 등 기본 데이터 셋 제공
2. Multi-Scale Crop function (Optional) 추가 -> IPYNB 구현 완료, 추후 Test 단계 적용 예정
3. Dense Evaluation function (Optional) 추가
4. VGG 전체 점진적 학습 / Transfer Learning (Optional) 추가
5. 데이터 증강 파라미터화 + 기본 데이터 증강 추가
6. ResNet 구현


---
작업 내용
---
## 2025 05 26
1. Hydra 적용
2. Training 종료 후 레포트 출력
3. Eval 함수 작성