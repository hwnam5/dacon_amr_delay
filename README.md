# 스마트 창고 출고 지연 예측

[DACON 대회 링크](https://dacon.io/competitions/official/236696/overview/description)

AMR 기반 스마트 창고에서 향후 30분간 평균 출고 지연 시간을 예측합니다.

- 평가 지표: MAE
- 타깃: `avg_delay_minutes_next_30m`
- 현재 메인 파이프라인: **1D CNN encoder 임베딩 + CatBoost/LightGBM/XGBoost + 2단계 stacking**

## 실행 순서

```bash
myenv/bin/python phase1_cnn_encoder.py
myenv/bin/python phase2_full_pipeline.py
```

`phase1_cnn_encoder.py`가 CNN encoder를 붙인 base 모델 OOF/test 예측을 만들고, `phase2_full_pipeline.py`가 그 결과를 읽어 weighted ensemble, KNN layout TE, iterative pseudo-labeling, mega stacking을 수행합니다.

## 입력 파일

| 경로 | 용도 |
|---|---|
| `open/train.csv` | 원본 학습 데이터 |
| `open/test.csv` | 원본 테스트 데이터 |
| `open/layout_info.csv` | layout 메타 정보 |
| `open/sample_submission.csv` | 제출 형식 |
| `processed/train_imputed.csv` | 1D CNN encoder 입력용 imputed train |
| `processed/test_imputed.csv` | 1D CNN encoder 입력용 imputed test |
| `best_model_1dcnn.pt` | 학습된 1D CNN 가중치 |

스크립트는 파일 위치와 현재 작업 디렉터리를 모두 확인해 `open/train.csv`가 있는 프로젝트 루트를 자동으로 찾습니다.

## 전체 흐름

```text
open/train.csv + layout_info.csv
  -> 원본/파생 tabular 피처 생성
  -> best_model_1dcnn.pt의 head 제거, encode() 임베딩 추출
  -> 원본/파생 피처 + cnn_emb_0~31 concat
  -> CatBoost / LightGBM / XGBoost 5-fold OOF 생성
  -> Target Encoding CatBoost
  -> Layout Cluster CatBoost
  -> Pseudo-labeling CatBoost
  -> Phase1 stacking
  -> phase2_artifacts/*_cnnenc.npy
  -> Phase2 weighted / KNN layout TE / iterative pseudo / mega stacking
  -> submission_phase2_cnnenc.csv
```

## Phase1: `phase1_cnn_encoder.py`

### 1. 데이터 로드

`open/train.csv`, `open/test.csv`, `layout_info.csv`를 읽고 `layout_id` 기준으로 join합니다.

각 `scenario_id`는 25개 time slot으로 구성되며, 시나리오 내부 위치를 `slot_idx`로 만듭니다.

### 2. 기본 피처 생성

주요 파생 피처:

| 분류 | 예시 |
|---|---|
| 로봇 상태 비율 | `idle_ratio`, `charging_ratio`, `active_ratio` |
| 이벤트 flag | `has_collision`, `has_fault`, `pack_saturated` |
| 상호작용 | `pack_x_slot`, `pack_x_battery`, `cong_x_inflow` |
| 누적값 | `cum_orders`, `cum_faults` |
| lag | `order_inflow_15m_lag1`, `pack_utilization_lag1` |
| layout 비율 | `robots_per_pack`, `robots_per_charger`, `area_per_robot` |
| 결측 flag | `battery_mean_isna`, `low_battery_ratio_isna` |
| scenario meta | 주요 피처의 scenario별 `mean/std/max/min` |

모델 학습은 `log1p(target)` 공간에서 수행하고 예측은 `expm1`로 복원합니다.

### 3. 1D CNN encoder 임베딩

`best_model_1dcnn.pt`를 로드하지만 예측 head는 사용하지 않습니다. `encode()` 출력만 사용합니다.

```text
processed/train_imputed.csv
processed/test_imputed.csv
  -> CNN 입력 피처 생성
  -> (scenario, 25 slots, features)
  -> model.encode()
  -> cnn_emb_0 ~ cnn_emb_31
```

이 32차원 임베딩을 Phase1 tabular 피처에 concat합니다.

```text
원본/파생 피처 + cnn_emb_0~31
```

### 4. Base 모델 3종

모든 모델은 `GroupKFold(n_splits=5)`를 사용하며 group은 `scenario_id`입니다.

| 모델 | 특징 | 출력 |
|---|---|---|
| CatBoost | `layout_id`, `layout_type` categorical 사용, 3 seed 평균 | `oof_cb_avg_cnnenc.npy` |
| LightGBM | `regression_l1`, MAE 최적화 | `oof_lgb_cnnenc.npy` |
| XGBoost | `reg:absoluteerror`, category는 코드값으로 변환 | `oof_xgb_cnnenc.npy` |

이미 `.npy`가 있으면 재학습하지 않고 로드합니다.

### 5. Target Encoding + CatBoost

`layout_id`, `layout_type`을 target 평균 기반 숫자 피처로 바꿉니다.

```text
layout_id -> layout_id_te
layout_type -> layout_type_te
```

validation leakage를 막기 위해 fold별로 validation fold를 제외한 train fold에서만 encoding을 계산합니다.

출력:

```text
oof_cb_te_cnnenc.npy
test_cb_te_cnnenc.npy
```

### 6. Layout Cluster + CatBoost

test에는 train에서 보지 못한 layout이 등장합니다. 이를 보완하기 위해 layout 구조 수치로 KMeans cluster를 만듭니다.

추가 피처:

```text
layout_cluster_5, layout_cluster_10, layout_cluster_20
layout_cluster_5_te, layout_cluster_10_te, layout_cluster_20_te
```

출력:

```text
oof_cb_te_cluster_cnnenc.npy
test_cb_te_cluster_cnnenc.npy
```

### 7. Pseudo-labeling + CatBoost

`CB + TE + Cluster`의 test 예측 중 신뢰 구간만 pseudo label로 사용합니다.

```text
5분 <= prediction <= 40분
```

출력:

```text
oof_cb_pseudo_cnnenc.npy
test_cb_pseudo_cnnenc.npy
```

### 8. Phase1 stacking

Base 예측값과 일부 meta 피처를 LightGBM meta model에 넣습니다.

```text
cb, lgb, xgb, cb_te, cb_te_cluster, cb_pseudo
+ low_battery_ratio_sc_mean
+ pack_utilization_sc_mean
+ congestion_score_sc_mean
+ slot_idx
+ pack_x_battery
```

출력:

```text
submission_phase1_cnn_encoder.csv
```

## Phase2: `phase2_full_pipeline.py`

Phase2는 Phase1의 `*_cnnenc.npy` 결과를 읽어 최종 앙상블을 만듭니다. Phase1에서 이미 CNN encoder를 사용했으므로 Phase2 내부 CNN 재학습은 기본적으로 꺼져 있습니다.

```python
RUN_PHASE2_CNN_RETRAIN = False
```

### 1. Phase1 결과 로드

읽는 파일:

```text
oof_cb_avg_cnnenc.npy / test_cb_avg_cnnenc.npy
oof_lgb_cnnenc.npy / test_lgb_cnnenc.npy
oof_xgb_cnnenc.npy / test_xgb_cnnenc.npy
oof_cb_te_cnnenc.npy / test_cb_te_cnnenc.npy
oof_cb_te_cluster_cnnenc.npy / test_cb_te_cluster_cnnenc.npy
oof_cb_pseudo_cnnenc.npy / test_cb_pseudo_cnnenc.npy
```

### 2. Weighted ensemble

Phase1 base 모델 예측값의 가중 평균을 grid search로 찾습니다.

출력:

```text
oof_weighted_cnnenc.npy
test_weighted_cnnenc.npy
```

### 3. CNN 재학습 스킵

기존 Phase2 노트북에는 CNN을 새로 학습하는 단계가 있었지만 현재 구조에서는 중복입니다.

```text
Phase1에서 이미 best_model_1dcnn.pt encoder 임베딩을 사용
-> Phase2 CNN retrain 스킵
```

### 4. KNN Layout Target Encoding

layout 구조가 비슷한 train layout을 KNN으로 찾아 target 평균을 피처로 추가합니다.

```text
layout_knn_te_k3
layout_knn_te_k5
layout_knn_te_k10
```

출력:

```text
oof_cb_knn_cnnenc.npy
test_cb_knn_cnnenc.npy
```

### 5. Iterative pseudo-labeling

Phase1 `cb_pseudo` 결과를 시작점으로 pseudo-labeling을 2회 반복합니다.

```text
iter1: 5~35분 test 예측을 pseudo label로 사용
iter2: 5~30분 test 예측을 pseudo label로 사용
```

출력:

```text
oof_pseudo_iter1_cnnenc.npy
test_pseudo_iter1_cnnenc.npy
oof_pseudo_iter2_cnnenc.npy
test_pseudo_iter2_cnnenc.npy
```

### 6. Mega stacking

모든 base 예측값과 Phase2 추가 모델을 LightGBM meta model에 넣습니다.

```text
cb_orig, lgb, xgb, cb_te, cb_cluster, cb_pseudo
weighted, cb_knn, cb_pseudo_iter1, cb_pseudo_iter2
+ scenario meta 5개
```

출력:

```text
oof_mega_cnnenc.npy
test_mega_cnnenc.npy
```

### 7. 최종 제출

weighted ensemble과 mega stacking 중 OOF MAE가 더 낮은 결과를 선택합니다.

출력:

```text
submission_phase2_cnnenc.csv
```

## 산출물 위치

| 경로 | 설명 |
|---|---|
| `phase2_artifacts/*.npy` | OOF/test 예측, CNN embedding, Phase2 중간 결과 |
| `submission_phase1_cnn_encoder.csv` | Phase1 stacking 제출 파일 |
| `submission_phase2_cnnenc.csv` | 최종 Phase2 제출 파일 |

## 핵심 아이디어

1. **GroupKFold 필수**: 같은 `scenario_id`가 train/validation에 섞이면 시계열 누수가 생깁니다.
2. **CNN encoder는 시간 흐름 압축기**: 25-slot 시퀀스 패턴을 32차원 임베딩으로 변환합니다.
3. **Target Encoding은 layout의 지연 성향을 숫자로 제공**합니다.
4. **Layout Cluster/KNN TE는 unseen layout 일반화용**입니다.
5. **Phase2는 Phase1 결과를 재조합하고 보강**합니다.
