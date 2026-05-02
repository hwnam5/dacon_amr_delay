# 스마트 창고 출고 지연 예측

[DACON 대회 링크](https://dacon.io/competitions/official/236696/overview/description)

AMR(자율이동로봇) 기반 스마트 창고에서 향후 30분간 평균 출고 지연 시간을 예측하는 AI 모델 개발

- **평가 지표**: MAE (낮을수록 좋음)
- **타깃**: `avg_delay_minutes_next_30m`

---

## 데이터

| 파일 | 설명 |
|---|---|
| `open/train.csv` | 학습 데이터 (10,000 시나리오 × 25행) |
| `open/test.csv` | 테스트 데이터 |
| `open/layout_info.csv` | 창고 레이아웃 정보 |
| `open/sample_submission.csv` | 제출 양식 |

피처 그룹 A~J (90개): 주문 유입, 로봇 상태, 배터리, 혼잡도, 패키징, 장애, 인력, IT 시스템, 환경, 공간

---

## 파이프라인

```
1. MICE 결측치 대체   python impute_mice.py
2. 모델 학습 / 추론   python model.py
```

결과물: `submission.csv`

---

## 모델 구조

```
그룹 A~J 피처
  → GroupMLP (residual) × 10
  → CLS 토큰 prepend
  → [TransformerEncoderLayer → FiLM] × 4   ← layout 조건부 변환
  → CLS 토큰 → Head → 예측값 (log 공간)
  → expm1 복원 → 분 단위 MAE
```

- **FiLM**: 창고 레이아웃(layout_type + 13개 수치 피처)으로 토큰을 조건부 변환
- **타깃 변환**: `log1p` 학습 → `expm1` 복원

---

## 코드 설명

### `impute_mice.py` — 결측치 대체

`IterativeImputer + LightGBM` 조합으로 MICE(Multiple Imputation by Chained Equations)를 수행합니다.

- 전체 수치형 컬럼(90개, TARGET 제외)에 단일 `IterativeImputer`를 적용
- **동작 방식**: 결측 열마다 나머지 모든 열을 feature로 LightGBM을 학습 → 예측으로 채움. 이 과정을 10라운드 반복하며 수렴
- train에서 `fit_transform`, test에서 `transform`만 적용해 데이터 누수 방지
- 결과: `processed/train_imputed.csv`, `processed/test_imputed.csv`

---

### `model.py` — 모델 학습 및 추론

#### WarehouseDataset
- `train_imputed.csv` + `layout_info.csv`를 `layout_id` 기준으로 merge
- 그룹 A~J 피처와 layout 수치 피처를 각각 `StandardScaler`로 정규화 (train fit → val/test transform)
- 타깃에 `log1p` 변환 적용 (skew=5.68 보정)
- val split: `scenario_id` 단위 80/20 랜덤 분리 (같은 scenario가 train/val에 섞이지 않도록)

#### GroupMLP
90개 피처를 도메인별 10개 그룹(A~J)으로 나눠 각각 독립적인 MLP로 인코딩합니다.

```
input(n) → Linear → LayerNorm → GELU → Linear → LayerNorm   (+proj shortcut)
         → GELU  → Linear → LayerNorm                        (+identity shortcut)
→ d_model 차원 토큰
```

그룹별로 독립 인코딩하는 이유: 성격이 다른 피처(로봇 상태 vs 환경 센서 등)를 같은 공간에 섞으면 간섭이 생길 수 있기 때문입니다.

#### LayoutEncoder
창고 레이아웃 정보를 조건 벡터로 인코딩합니다.

```
layout_type(범주형) → Embedding(4, 8)  ─┐
layout_nums(13개 수치) ─────────────────┴→ Linear → LayerNorm → GELU → Linear → d_layout
```

#### FiLM (Feature-wise Linear Modulation)
layout 조건 벡터로 각 토큰을 아핀 변환합니다.

```
output = γ(layout_emb) × token + β(layout_emb)
```

좁은 통로 창고(narrow)와 허브앤스포크 창고(hub_spoke)는 같은 혼잡도 수치가 다른 의미를 가질 수 있습니다. FiLM은 layout에 따라 피처의 스케일과 편향을 동적으로 조정해 이를 반영합니다.

#### Transformer + CLS 토큰
10개 그룹 토큰 앞에 학습 가능한 CLS 토큰을 붙여 Transformer에 통과시킵니다.

```
[CLS, 토큰A, 토큰B, ..., 토큰J]  →  [TransformerLayer → FiLM] × 4
```

- CLS 토큰이 Attention을 통해 중요한 그룹에 선택적으로 집중
- 각 Transformer 레이어 직후 FiLM을 적용해 layout 조건을 레이어마다 반영
- 최종적으로 `tokens[:, 0]` (CLS 위치)만 Head에 전달

#### Head
```
CLS 토큰 → LayerNorm → Linear(128→64) → GELU → Dropout
         → Linear(64→16) → GELU → Dropout → Linear(16→1) → Softplus
```

`Softplus`로 출력값을 양수로 제한 (지연 시간은 음수가 될 수 없음). 추론 후 `expm1`으로 log 공간에서 분 단위로 복원합니다.

#### 학습 전략
| 항목 | 내용 |
|---|---|
| Loss | `L1Loss` (MAE) — 평가 지표와 직접 정렬 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-3) |
| Scheduler | LinearLR warmup 10 epoch → CosineAnnealingLR 490 epoch |
| Early stopping | val MAE 기준 patience=20 |

---

## 주요 설정

| 항목 | 값 |
|---|---|
| d_model | 128 |
| Transformer layers | 4 |
| Optimizer | AdamW (lr=1e-3) |
| Scheduler | Warmup 10 epoch + CosineAnnealing |
| Early stopping | patience=20 |
| Val split | scenario 단위 80/20 |
