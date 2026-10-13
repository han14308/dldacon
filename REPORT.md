# 합성 미세구조 경도 예측 — 진행 정리 (2026-10-13)

대회: [DAKER / 데이콘 합성 미세구조 기반 재료 물성 예측](https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai)
입력 256×256 8bit 흑백 PNG 1장 → 경도 대용값(HV) 회귀, 평가 RMSE. Train 500 / Test 1,000, Public = test 30%.
제출 마감 10/19 10:00, 하루 5회. 리더보드 1위 8.84, 20위 10.07.

## 1. 결과 요약

| 구분 | 모델 | Train CV RMSE (5-fold) | Public |
|---|---|---|---|
| 기준 | 평균값 예측 (194.17) | 17.68 | **16.57** |
| A | 물리 특징 134개 + LightGBM(huber) | 14.44 | |
| A | 물리 특징 166개 + LightGBM(L2) | 14.14 | |
| A | + 열화 증강 행 2,000개 | 13.92 | |
| B | NASA MicroNet ResNet50 fine-tune | 14.07 | |
| C | DINOv2 ViT-S/14 (252) 고정 + Ridge | 14.00 | |
| C | DINOv2 ViT-S/14 (518, 2배 업스케일) + Ridge | 14.29 | |
| C | ConvNeXt-Tiny (in22k) fine-tune | 13.80 | |
| 독립 | torchvision ResNet18, 224, MSE, 플립만 (공통 코드 없음) | 13.92 | |
| 앙상블 | A v2 + B + C(DINO) + C(ConvNeXt) NNLS 스태킹 | **13.28** | **12.13** |

제출 파일: `submissions/stack_final.csv` (Public 12.13), `submissions/const_mean.csv` (Public 16.57).

중단된 실험(결과 없음): ConvNeXt + 열화 증강 + [원본, 디노이즈, 국소표준편차] 3채널 입력 / ConvNeXt 512 업스케일 / ConvNeXt 160 랜덤 크롭 + 3×3 격자 TTA. 코드에는 옵션(`--degrade --chan3 --upscale --crop --mixup`)으로 남아 있어 재실행 가능.

## 2. 데이터 관찰

- 타깃: 평균 194.2, 표준편차 17.7, 최소 152, 최대 264 (오른쪽 꼬리). 단봉 연속 분포.
- 이미지: Voronoi 결정립 + 어두운 2상 알갱이 + 검은 기공 + 신장/정렬 + 블러/노이즈/대비 교란.
- 단일 특징 Spearman 상관(절댓값 상위): 어두운 픽셀 비율 +0.52, 기공 면적 −0.46, 첨도 −0.50, 자기상관 길이 +0.33(결정립이 **굵을수록** 단단: Hall–Petch와 반대 방향), 신장도 +0.30. 노이즈·블러 지표는 ~0 (교란 변수).
- LightGBM 상위 특징: 어두운 상 비율 > 결정립 정렬도 > 기공 면적 > GLCM 이방성.

## 3. 검증한 가설과 결론

| # | 가설 | 검증 방법 | 결과 |
|---|---|---|---|
| 1 | Huber 손실이 극단값을 평균으로 당긴다 | LightGBM huber vs L2 | 14.25 vs 14.22, 영향 없음 |
| 2 | 타깃 노이즈 한계 ~13 | 잔차 상관 분석 | 4개 모델 잔차 상관 0.83~0.90, 잔차-타깃 상관 0.83. 그러나 리더보드가 8.8이라 노이즈 한계 아님 |
| 3 | 결정립 절대 방향이 경도에 영향 | 구조텐서 각도별 평균 경도, 각도 특징 추가 | 각도 구간 평균 197~205로 차이 없음. 기각 |
| 4 | 배율 미제공 → 기공 크기로 스케일 보정 | 결정립/기공 크기 비율 특징 | 14.14→14.12, 기각 |
| 5 | 원본 밝기 레벨·스펙트럼에 숨은 신호 | 히스토그램 256, 방사/각 PSD, 고유 레벨 수 → Ridge | 15.7~16.4, 기각 |
| 6 | train 라벨 일부 오염(섞임) | 잔차 2-GMM, 첨도, 트리밍 학습 | 첨도 3.8, 혼합 분리 안 됨. 기각 |
| 7 | test가 train 재렌더링(시드 공유) | DINO 코사인, 8대칭 NCC, 위상상관(이동 포함) | test→train 매칭 분포가 train 내부와 동일. 기각 |
| 8 | train/test 이미지 분포 차이 | adversarial validation | AUC 0.46, 동일 분포 |
| 9 | test 정답 분포가 더 좁다 | 상수 제출 | 16.57 → train(17.7)과 사실상 동일 |
| 10 | 화질 교란이 오차의 주원인 | 노이즈/블러 사분위별 오차 | 깨끗한 이미지 11.6 vs 노이즈 이미지 14.3~15.0. **유일하게 확인된 양성 신호** |
| 11 | 제 파이프라인 버그 | 공통 코드 없는 독립 베이스라인 | 13.92로 동일. 버그 아님 |
| 12 | 해상도(1px 경계선이 stem에서 소실) | DINOv2 518 | 14.29, ViT에서는 효과 없음. CNN 512는 미완 |

종합: 데이터 누수·분포 차이·라벨 오염 류는 모두 기각. 상수 제출(16.57)과 상위권(8.8~10)을 대수적으로 맞춰 보면 상위권 모델은 train CV로도 약 11.5가 나와야 하며, 현재 13.3과의 차이는 **신호 추출력(모델링) 격차**다. CV가 여전히 올바른 나침반이다.

## 4. 코드 구조

```
src/common.py          경로, 고정 fold(타깃 10분위 층화, seed 42), 이미지 로딩, RMSE
src/features.py        트랙 A 특징 166개 (화질/스펙트럼/구조텐서/GLCM/결정립/2상/기공/LBP/평탄영역/자기상관 모멘트)
track_a_features.py    특징 추출 → outputs/features_{train,test}.csv
track_a_augment.py     열화(블러/노이즈/대비/감마) 복제본 4개의 특징 → features_train_aug.csv
track_a_gbdt.py        LightGBM 5-fold × seed (--aug, --objective)
train_cnn.py           트랙 B/C 공용 학습기: micronet_resnet50 또는 timm 백본
                       옵션: --degrade --chan3 --loss mse --upscale --crop --mixup, 8대칭 TTA
track_c_dino.py        DINOv2 임베딩(8대칭 TTA 평균) + Ridge (--img_size)
baseline_plain.py      공통 코드 없는 독립 ResNet18 베이스라인(버그 검증용)
stack.py               oof 기반 비음수 선형 스태킹(nested CV) → submissions/
```

실행 순서는 `README.md`. 사전학습 가중치: MicroNet ResNet50 (HF `jstuckner/microscopy-resnet50-micronet`, MIT), timm ImageNet/DINOv2 (Apache-2.0). 외부 데이터·test 학습 미사용.

환경: Windows 11, Python 3.12, RTX 4070 Laptop 8GB, torch 2.14+cu126 (`requirements.txt`).

## 5. 다음에 시도할 것 (우선순위)

1. **노이즈 강건성에 집중** (가설 10): 열화 증강을 강하게 건 CNN. 단, 디노이즈 채널은 학습 루프 밖에서 1회 캐싱할 것 (매 epoch NLM 호출은 CPU 25시간을 먹고 중단됨).
2. ConvNeXt 512 업스케일, 160 크롭 + 격자 TTA, mixup — 코드는 준비돼 있음. 각각 CV 12 이하가 나오는지 확인.
3. 생성기 역설계: Voronoi + 2상 + 기공 + 신장 + 열화를 직접 생성해 잠재변수 회귀기를 무한 데이터로 학습한 뒤, 그 출력을 특징으로 사용. 규칙상 "외부 데이터" 해석이 애매하니 게시판 문의 후 진행.
4. 앙상블 확대(seed 3개, EfficientNet 추가)는 0.2~0.5 개선에 그치므로 1~3 뒤에.
