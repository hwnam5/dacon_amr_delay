# AMR 스마트 물류창고 출고 지연 예측 — Kim_JY 접근법

## 전체 파이프라인

```
train.csv + layout_info.csv
  └─ layout join (layout_id 기준)
  └─ slot_idx 추가 (시나리오 내 시간 위치 0~24)
  └─ 피처 엔지니어링 build_features()   94 → 155개
  └─ 시나리오 메타 피처 add_scenario_meta()  155 → 195개
  └─ 학습 피처 선택 (ID, scenario_id, target 제외) → 192개
  └─ LightGBM 5-Fold GroupKFold (by scenario_id)
  └─ (선택) 분위수 앙상블 P50 + P90
  └─ submission.csv / submission_ensemble.csv
```

---

## 데이터 기본 정보

| 항목 | 값 |
|---|---|
| train shape | (250,000, 94) |
| test shape | (50,000, 93) |
| layout shape | (300, 15) |
| train layout 수 | 250개 |
| test layout 수 | 100개 |
| 신규 layout (test에만 등장) | **50개** |
| 타깃 평균 | 18.96분 |
| 타깃 표준편차 | 27.35분 |
| 타깃 최대 | 715.86분 |

신규 layout 50개는 train에서 본 적 없는 창고 구조 → layout_id 자체가 강한 피처이면서 동시에 일반화 위험 요소

---

## 전처리

### 1. layout_info join
```python
train = train.merge(layout, on='layout_id', how='left')
```
aisle_width_avg, charger_count, robot_total 등 13개 수치 + layout_type 추가

### 2. slot_idx 추가
시나리오 안에서 슬롯의 시간 위치(0~24)를 나타내는 인덱스

```python
df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
df['slot_idx'] = df.groupby('scenario_id').cumcount()
```

### 3. 결측치 처리: 별도 대체 없음
- LightGBM이 NaN을 트리 분기에서 최적 방향으로 자동 처리
- 결측 여부 자체는 `_isna` 플래그로 명시적 인코딩

### 4. 카테고리 변환
`layout_id`, `layout_type` → `category` dtype → LightGBM이 자동 인식

### 5. 타깃 변환
```python
y_log = np.log1p(y_train)   # 학습
pred  = np.expm1(pred_log)  # 복원
```
heavy-tail 완화 (skew 5.68 → 0.08)

---

## 피처 엔지니어링

`build_features()` 함수로 train/test에 동일하게 적용

### Tier 1: 임계점 Dummy

| 피처 | 조건 | 의미 |
|---|---|---|
| `pack_saturated` | pack_utilization > 0.95 | 패킹 포화 |
| `pack_idle_anomaly` | pack_util < 0.1 AND slot_idx > 5 | 초반 이후 이상 유휴 |
| `low_battery_alert` | low_battery_ratio > 0.034 | 저배터리 경보 |
| `battery_critical` | battery_mean < 58 | 배터리 위험 수준 |
| `density_alert` | max_zone_density > 0.05 | 밀도 경보 |

### Binary Indicator (대부분 0인 이벤트 변수)

| 피처 | 원본 변수 |
|---|---|
| `has_collision` | near_collision_15m > 0 |
| `has_block` | blocked_path_15m > 0 |
| `has_fault` | fault_count_15m > 0 |
| `has_reassign` | task_reassign_15m > 0 |

### 비율 변수 (★ 단일 변수 중 가장 강한 신호)

```python
total_fleet   = robot_active + robot_idle + robot_charging
idle_ratio    = robot_idle     / (total_fleet + 1e-6)
charging_ratio = robot_charging / (total_fleet + 1e-6)
active_ratio  = robot_active   / (total_fleet + 1e-6)
```

### Tier 2: 시간 상호작용

| 피처 | 계산식 | 의미 |
|---|---|---|
| `pack_x_slot` | pack_utilization × slot_idx | 시간 경과에 따른 패킹 누적 압력 |
| `order_x_slot` | order_inflow_15m × slot_idx | 시간 경과에 따른 주문 누적 압력 |
| `inflow_x_urgent` | order_inflow_15m × urgent_order_ratio | 급박한 주문 부하 |
| `effective_load` | order_inflow_15m × (1 + 0.5 × urgent_ratio) | 가중 유효 부하 |

### Cascading 곱항 (두 병목 동시 발생)

- `pack_x_battery` = pack_utilization × low_battery_ratio
- `pack_x_charging` = pack_utilization × charging_ratio
- `inflow_x_battery` = order_inflow_15m × low_battery_ratio
- `cong_x_inflow` = congestion_score × order_inflow_15m

### 누적(Stock) 변수

```python
cum_orders     = cumsum(order_inflow_15m)    by scenario_id
cum_collisions = cumsum(near_collision_15m)  by scenario_id
cum_faults     = cumsum(fault_count_15m)     by scenario_id
cum_unique_sku = cumsum(unique_sku_15m)      by scenario_id
```

### Lag 피처

대상: `order_inflow_15m`, `pack_utilization`, `low_battery_ratio`, `congestion_score`

```python
f_lag1  = groupby(scenario_id)[f].shift(1)
f_diff1 = f - f_lag1
```

→ 각 피처 × (lag1 + diff1) = 8개 추가

### 결측 플래그

```python
for col in ['battery_mean', 'low_battery_ratio', 'congestion_score',
            'pack_utilization', 'order_inflow_15m']:
    df[f'{col}_isna'] = df[col].isna().astype(int)
```

### Layout 파생 피처

```python
robots_per_pack    = robot_total / pack_station_count   # 패킹 당 로봇 수
robots_per_charger = robot_total / charger_count        # 충전기 당 로봇 수
area_per_robot     = floor_area_sqm / robot_total       # 로봇 당 면적
pack_per_charger   = pack_station_count / charger_count
```

Layout type 상호작용 (좁은 창고에서 배터리 위험 증폭):
```python
is_narrow          = (layout_type == 'narrow')
narrow_x_battery   = is_narrow × low_battery_ratio
few_packs          = (pack_station_count < 7)
few_packs_x_util   = few_packs × pack_utilization
```

---

## 시나리오 메타 피처 (★★★ 가장 강력)

> **between-scenario 변동이 within-scenario 변동의 3.3배**

각 시나리오의 25개 슬롯에서 통계량(mean/std/max/min)을 계산해 모든 행에 broadcast.
Top 피처 대부분을 차지.

```python
key_features = [
    'order_inflow_15m', 'pack_utilization', 'low_battery_ratio',
    'congestion_score', 'urgent_order_ratio', 'idle_ratio', 'charging_ratio',
    'robot_charging', 'fault_count_15m', 'staff_on_floor',
]
# 10개 피처 × 4 통계 = 40개 메타 피처
```

### 피처 개수 변화

| 단계 | shape |
|---|---|
| 원본 | (250,000, 94) |
| layout join + slot_idx | (250,000, 109) |
| build_features() | (250,000, 155) |
| add_scenario_meta() | (250,000, 195) |
| 최종 학습 피처 수 | **192개** |

---

## 모델: LightGBM 5-Fold GroupKFold

### 하이퍼파라미터

| 항목 | 값 |
|---|---|
| objective | `regression` (RMSE) |
| metric | `rmse` |
| learning_rate | 0.05 |
| num_leaves | 63 |
| min_child_samples | 100 |
| feature_fraction | 0.7 |
| bagging_fraction | 0.8 |
| bagging_freq | 5 |
| reg_alpha | 0.1 |
| reg_lambda | 0.1 |
| num_boost_round | 3000 (early stopping 100) |

GroupKFold 기준: `scenario_id` → 같은 시나리오가 train/val에 섞이지 않음

---

## Val 성능 (OOF)

| Fold | RMSE | MAE | best_iter |
|---|---|---|---|
| 1 | 23.6201 | 8.8656 | 95 |
| 2 | 23.0569 | 9.1447 | 188 |
| 3 | 21.2229 | 8.6021 | 88 |
| 4 | 24.5331 | 9.4261 | 87 |
| 5 | 22.0647 | 8.9600 | 92 |
| **전체 OOF** | **22.9289** | **8.9997** | — |
| Fold 편차 | ±1.2962 | ±0.3084 | — |

### 잔차 진단 (target 구간별 MAE)

| 구간 | 샘플 수 | 실제 평균 | MAE | P90 오차 |
|---|---|---|---|---|
| 0–5분 | 75,049 | 2.73 | 2.59 | 4.62 |
| 5–15분 | 82,118 | 8.66 | 3.83 | 7.74 |
| 15–30분 | 40,732 | 22.07 | 9.22 | 17.28 |
| 30–60분 | 39,186 | 41.35 | 11.74 | 24.88 |
| **60–120분** | 10,370 | 78.31 | **47.18** | 75.36 |
| **120분+** | 2,545 | 193.66 | **163.64** | 269.53 |

Burst 영역(60분+)이 RMSE의 주된 원인. 0~15분 구간은 잘 맞춤.

---

## Top 30 피처 중요도 (5-Fold Gain 합계)

| 순위 | 피처 | Gain | 분류 |
|---|---|---|---|
| 1 | charging_ratio_sc_mean | 1,193,292 | Scenario Meta |
| 2 | congestion_score_sc_mean | 1,000,545 | Scenario Meta |
| 3 | layout_id | 789,367 | Layout |
| 4 | robot_charging_sc_mean | 649,154 | Scenario Meta |
| 5 | low_battery_ratio_sc_mean | 644,154 | Scenario Meta |
| 6 | pack_utilization_sc_mean | 643,454 | Scenario Meta |
| 7 | low_battery_ratio_sc_std | 374,762 | Scenario Meta |
| 8 | charging_ratio | 152,312 | Ratio |
| 9 | pack_utilization_sc_min | 141,497 | Scenario Meta |
| 10 | congestion_score_sc_std | 134,966 | Scenario Meta |
| 11 | charging_ratio_sc_max | 133,980 | Scenario Meta |
| 12 | pack_utilization | 133,334 | — |
| 13 | order_inflow_15m_sc_mean | 95,739 | Scenario Meta |
| 14 | charging_ratio_sc_std | 87,503 | Scenario Meta |
| 15 | slot_idx | 71,348 | — |
| 16 | avg_trip_distance | 69,977 | — |
| 17 | pack_utilization_sc_std | 62,958 | Scenario Meta |
| 18 | order_inflow_15m_sc_max | 57,651 | Scenario Meta |
| 19 | max_zone_density | 47,720 | — |
| 20 | congestion_score | 43,607 | — |
| 21 | order_inflow_15m_sc_std | 42,846 | Scenario Meta |
| 22 | robots_per_pack | 29,733 | Layout |
| 23 | pack_x_slot | 28,466 | Interaction |
| 24 | pack_station_count | 28,402 | Layout |
| 25 | order_inflow_15m_sc_min | 26,738 | Scenario Meta |
| 26 | idle_ratio_sc_max | 25,557 | Scenario Meta |
| 27 | idle_ratio_sc_std | 22,269 | Scenario Meta |
| 28 | cum_collisions | 21,106 | Stock |
| 29 | avg_items_per_order | 20,926 | — |
| 30 | congestion_score_sc_min | 17,971 | Scenario Meta |

Top 30 중 시나리오 메타(`_sc_*`) 19개, Layout 관련 3개, 비율 1개, 상호작용 1개, 누적 1개

---

## 분위수 앙상블 (선택 섹션 12~13)

3가지 목적함수로 각각 5-fold 학습 후 가중 앙상블:

| 모델 | objective | OOF RMSE | OOF MAE | 특징 |
|---|---|---|---|---|
| Mean | regression (RMSE) | 22.9289 | 8.9997 | 기본 예측 |
| P50 | quantile α=0.5 | 22.5453 | 8.9772 | median 예측, 다른 손실함수로 다양성 |
| P90 | quantile α=0.9 | 26.1430 | 14.5299 | burst 영역 특화 |

### OOF grid search 최적 가중치

| 기준 | Mean | P50 | P90 | RMSE | MAE |
|---|---|---|---|---|---|
| RMSE 최소화 | 0.40 | 0.25 | 0.35 | **21.0716** | 9.6877 |
| **MAE 최소화** | **0.50** | **0.45** | **0.05** | 22.2708 | **8.9361** |

대회 평가 지표가 MAE이므로 MAE 최소화 가중치 채택 → `submission_ensemble.csv`

---

## 출력 파일

| 파일 | 설명 |
|---|---|
| `submission.csv` | Mean 모델 5-fold 앙상블 |
| `submission_ensemble.csv` | Mean(0.5) + P50(0.45) + P90(0.05) 앙상블 |
| `oof_pred.npy` | OOF 예측 (raw scale) |
| `test_pred.npy` | test 예측 (raw scale) |

### 제출 예측 통계 (submission.csv)

| min | max | mean | median |
|---|---|---|---|
| 0.00분 | 81.21분 | 17.72분 | 12.64분 |

---

## 핵심 인사이트

1. **시나리오 메타가 압도적** — Top 30 중 19개. between-scenario 변동이 within보다 3.3배 큼
2. **충전 병목이 1위** — `charging_ratio_sc_mean` gain 1위. 로봇 충전 병목 = 지연의 핵심 원인
3. **layout_id가 3위** — test 신규 layout 50개가 있어 일반화 중요
4. **Burst 영역 어려움** — 60분+ 구간 MAE 47~164. 분위수 앙상블로 부분 보완 (MAE 8.9997 → 8.9361)
5. **결측치 처리 불필요** — LightGBM 자체 NaN 처리 활용, 별도 imputation 없음
6. **GroupKFold 필수** — scenario_id 기준 분리 없으면 같은 시나리오 내 데이터 누설 발생
