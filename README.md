# 합성 미세구조 경도 예측 (DAKER / 데이콘)

환경: Windows 11, Python 3.12, RTX 4070 Laptop 8GB. 라이브러리 버전은 `requirements.txt`.
데이터는 `open/` (train/, test/, train.csv, sample_submission.csv) 상대경로로 읽는다.

## 실행 순서
```
.venv\Scripts\python track_a_features.py        # 트랙 A: 물리 특징 134개 추출 -> outputs/features_*.csv
.venv\Scripts\python track_a_gbdt.py            # 트랙 A: LightGBM 5-fold x 5 seed
.venv\Scripts\python track_c_dino.py            # 트랙 C: DINOv2 ViT-S/14 임베딩 + Ridge
.venv\Scripts\python train_cnn.py --backbone micronet_resnet50          # 트랙 B: NASA MicroNet ResNet50 (MIT)
.venv\Scripts\python train_cnn.py --backbone convnext_tiny.fb_in22k_ft_in1k   # 트랙 C: ConvNeXt-T
.venv\Scripts\python stack.py                    # 모든 oof 결합 -> submissions/stack.csv
```
각 스크립트는 `outputs/oof_<tag>.csv`, `outputs/test_<tag>.csv`, `submissions/<tag>.csv`를 남긴다.
fold는 `outputs/folds.csv`에 고정(타깃 10분위 층화, seed 42).

## 사전학습 가중치 (규칙 2-1 준수)
- MicroNet ResNet50: https://huggingface.co/jstuckner/microscopy-resnet50-micronet (MIT)
- timm ImageNet 가중치 (Apache-2.0), DINOv2 (Apache-2.0)
외부 데이터 미사용, test 이미지는 학습에 미사용.
