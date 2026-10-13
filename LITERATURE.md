# 관련 논문 조사 및 모델 선정 (2026-10-13)

목적: train 500장, 합성 Voronoi 미세구조 이미지 → 경도 회귀, 화질 교란(블러·노이즈·대비) 존재. 현재 CV 13.3 / Public 12.13, 리더보드 1위 8.84. 어떤 모델·학습 레시피가 맞는지 최신 논문에서 찾는다.

## 1. 핵심 논문

### 1.1 합성 미세구조 → 물성 회귀, 대회와 거의 같은 설정
**Predicting Steel Fatigue Life from Micrographs Using Physics-Informed Deep Learning** — [arXiv 2607.28695 (2026)](https://arxiv.org/abs/2607.28695)

- 생성기: Voronoi 결정립 30~120개, 로그정규 크기분포, 포아송 기공(반지름 U[1,8] px), 타원형 2상 입자(종횡비 U[1.2,3.0]), 조명·노이즈 열화. 데이콘 참고 PDF의 설명(결정립/2상/기공/신장/화질)과 구조가 같다.
- 라벨: 정규화된 기술자들의 **선형식 + 가우시안 노이즈**. 예) log10(Nf) = 7.0 − 1.2·균열 − 0.8·기공분율 − 0.5·log10(d/50+1) − 0.3·개재물밀도 + ε, ε~N(0,0.15). 결정립 항이 Hall–Petch형. 데이콘 라벨도 같은 구조일 가능성이 크고, 그렇다면 **노이즈 하한이 존재**한다.
- 결과(테스트 200장, R²): ResNet-50 0.93 > VGG-16 0.91 > SE-CNN 0.88 > 특징공학 28개 + GBM 0.86. **CNN fine-tune이 특징공학을 이긴다.**
- 레시피:
  - 입력 224, ImageNet 정규화, ResNet-50 stage 1~2 동결
  - AdamW 3e-4, cosine → 1e-6, 60 epoch, batch 16, early-stop patience 12, grad clip 1.0
  - 증강: 플립, 90° 회전, **shift-scale-rotate(±20°), elastic, 가우시안 노이즈, CLAHE, 밝기/대비**
  - 손실: **Gaussian NLL** (평균 μ와 log σ² 동시 예측). MSE 대비 ECE 76% 감소.
- 시사점: 우리가 의도적으로 피한 스케일 변형·강한 노이즈 증강을 전부 쓴다. GNLL은 노이즈 심한 이미지의 가중치를 모델이 스스로 낮추게 해 준다. 우리가 확인한 유일한 양성 신호(깨끗한 이미지 11.6 vs 노이즈 이미지 14~15)를 직접 공략하는 레시피.

### 1.2 파운데이션 ViT 특징으로 경도 예측
**Machine learning of microstructure–property relationships … foundational vision transformers** — [arXiv 2501.18637 (2025)](https://arxiv.org/abs/2501.18637), 선행 워크숍 논문 [CVPRW 2024 Whitman et al.](https://openaccess.thecvf.com/content/CVPR2024W/CV4MS/papers/Whitman_Learning_Microstructure--Property_Relationships_in_Materials_with_Robust_Features_from_Vision_CVPRW_2024_paper.pdf)

- CLIP(3종), DINOv2(S/B/L/g), SAM(H)의 **CLS 토큰** → PCA → 선형/2차/SVR.
- 실험 SEM 149장 비커스 경도: **DINOv2-L/14 + PCA 34성분 + SVR = MAPE 13.5%** (최고). MicroNet ResNet50 고정 특징 17.2%, 2점 상관 17.9%.
- 시사점: 우리는 ViT-S + Ridge(CV 14.0)만 시도. L 모델 + PCA + SVR(RBF)은 미시도 조합이며 30분 안에 검증 가능.

### 1.3 소량 라벨에서는 LoRA가 full fine-tune보다 유리
- [arXiv 2604.16083](https://arxiv.org/pdf/2604.16083): 라벨이 적으면 full fine-tune이 사전학습 prior를 덮어써 손해, 저랭크 어댑터가 보존.
- [arXiv 2508.21041, MIDOG 2025](https://arxiv.org/pdf/2508.21041): DINOv3-H+ + LoRA(1.3M 파라미터) + 강한 증강으로 큰 도메인 갭(자연영상→조직병리)에서도 전이 성공.
- [arXiv 2609.25567, RootQuantV2](https://arxiv.org/abs/2609.25567): 고정 DINOv3 ViT-L/16 + 어댑터(3.8% 파라미터)로 이미지 단위 회귀 R² 0.95. 상위 블록 해제가 성능을 올리지 못했다는 ablation.
- 시사점: 500장 규모에서는 DINOv2/v3 + LoRA 회귀가 맞는 적응 방식. timm에 DINOv3 ViT/ConvNeXt 가중치 포함.

### 1.4 백본 선택
- [arXiv 2406.05612](https://arxiv.org/pdf/2406.05612): 자원 제한 fine-tune 비교에서 **ConvNeXt-Tiny가 Swin/SwinV2-Tiny보다 거의 모든 데이터셋에서 우수**. 우리 결과(ConvNeXt-T 13.80이 단일 최고)와 일치.
- [arXiv 2510.07191](https://arxiv.org/pdf/2510.07191): ImageNet/DINOv2/DINOv3 초기화 차이는 작고 **입력 해상도가 초기화만큼 중요**. 우리 DINOv2 518 실험은 효과 없었으나 CNN 512는 미완.
- [MicroNet, npj Comput Mater 2022](https://www.nature.com/articles/s41524-022-00878-5): 현미경 10만장 사전학습 인코더. 1.2에서 고정 특징으로는 DINOv2에 밀렸고, 우리 fine-tune도 14.07로 ConvNeXt보다 낮았다.

### 1.5 기타
- [Springer 2026, 강재 ViT 물성 예측·데이터 누수](https://link.springer.com/article/10.1007/s10853-026-12405-8): 이미지를 패치로 잘라 학습·집계. EfficientNetV2와 ViT가 비슷(정규화 MAE 0.48 vs 0.50). 본문 접근 제한.
- [arXiv 2410.05762, GSNets](https://arxiv.org/pdf/2410.05762): 결정립 크기 등급 분류용 guided self-attention, Swin V2 대비 +1.9%p. 분류 문제이고 3,599장이라 직접 적용은 어려움.
- [arXiv 2601.08043](https://arxiv.org/html/2601.08043v1): 학습 데이터에 소량 노이즈 추가가 정규화로 작용해 입력 열화에 강해짐.

## 2. 추천 모델 (우선순위)

| 순위 | 모델 / 레시피 | 근거 | 예상 소요 (RTX 4070 8GB) |
|---|---|---|---|
| 1 | **ConvNeXt-Tiny(또는 ResNet-50) + 1.1 레시피**: 앞단 동결, Gaussian NLL, 강한 열화 증강(노이즈·CLAHE·밝기대비·shift-scale-rotate), AdamW 3e-4 cosine 60 epoch | 같은 설정의 논문에서 CNN이 최고, 화질 교란을 직접 공략 | 코드 10분 + 5-fold 학습 30~35분 + 예측/스태킹 5분 ≈ **45~50분** |
| 2 | **DINOv2-L/14(또는 DINOv3) CLS + PCA + SVR(RBF)** | 1.2에서 경도 예측 1위 조합, 미시도 | 약 30분 |
| 3 | **DINOv2/v3 + LoRA 회귀 헤드** | 1.3, 소량 라벨에 맞는 적응 | 약 1시간 |
| 4 | 1~3 스태킹 | 기존 4모델 스태킹이 단일 대비 −0.5 | 5분 |

주의: 증강은 GPU 텐서 연산으로 구현할 것. 이전에 샘플마다 CPU 디노이즈를 매 epoch 호출한 실험은 CPU 25시간을 먹고 중단됐다.

## 3. 목표치에 대한 함의

1.1의 라벨에는 N(0, 0.15) 노이즈가 명시적으로 들어간다(범위 5의 약 3%). 데이콘 라벨도 같은 구조라면 리더보드 8.8이 노이즈 하한 근처일 수 있다. 현실적 목표는 9~10 진입.
