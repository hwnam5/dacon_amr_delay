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

## 주요 설정

| 항목 | 값 |
|---|---|
| d_model | 128 |
| Transformer layers | 4 |
| Optimizer | AdamW (lr=1e-3) |
| Scheduler | Warmup 10 epoch + CosineAnnealing |
| Early stopping | patience=20 |
| Val split | scenario 단위 80/20 |
