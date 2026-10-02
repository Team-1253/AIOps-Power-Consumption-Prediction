# AIOps-Power-Consumption-Prediction

AI 데이터센터 전력 수요 예측 서비스 (팀명: 기가(GW) 막힐 땐).
최근 측정값으로 다음 1시간 전력 사용량(kWh)을 예측하는 LSTM 모델을 FastAPI로 서빙하고,
드리프트 감지 → fine-tuning 재학습 → MLflow 승격까지 대시보드에서 확인한다.

- 기반 코드: SKALA 실습 HAIC 스켈레톤 (`docs/HAIC_skeleton_README.md`)
- 프로젝트 개요: `docs/프로젝트_개요.md`
- 데이터: NREL ESIF HPC 데이터센터 실측값을 1시간 단위로 정제한 `data/hourly_clean.csv` (`data/README_dataset.md`)

## 데이터 형식 (API 계약)

입력 피처는 정제 CSV의 수치 컬럼을 그대로 쓴다. 요청 필드명 = CSV 컬럼명.

| 컬럼 | 용도 |
| --- | --- |
| `date_utc`, `time_utc` | 시간 (입력 피처 아님) |
| `energy_kwh` | 입력 피처 · 예측 대상 |
| `humi_pct` | 입력 피처 (외기 습도 %) |
| `temp_F` | 입력 피처 (외기 온도 °F) |

- 85,936행 중 71,372행만 세 값이 모두 있다. 나머지는 행 전체가 비어 있다(측정 누락).
- 누락 때문에 1시간 간격이 끊긴 구간이 802개다.

## 실행

```bash
pip install -r requirements.txt
uvicorn serving_app.main:app --host 0.0.0.0 --port 8077
```

대시보드 `http://localhost:8077/`, API 문서 `http://localhost:8077/docs`.
모델을 MLflow에서 불러올 때는 `MODEL_SOURCE=mlflow`, 서버 시작 시 바로 불러올 때는 `LOADING_MODE=eager`를 붙인다.

## 대시보드 (serving_app/static/index.html)

| 탭 | 내용 |
| --- | --- |
| 예측 관제 | 예측 실행, 배치 주입(정상 / 드리프트), 요청 지표 요약 |
| AI 대시보드 | 운영 지표 KPI, 처리 7단계, 재학습 이력, 현재 운영 모델, 최근 알람 |
| 데이터 | CSV 업로드, 데이터셋 통계, 데이터 출처 |
| 시스템 | 서버 설정값, 운영 로그 조회 |

- 드리프트 배치: 정상 배치와 같은 랜덤워크로 만들되 변동성(σ)을 3배로 키워 강제로 드리프트를 일으킨다.
- 입력 길이, 판정 윈도우, 입력 피처, 기준값은 `/system/info`에서 받는다. 화면 코드는 상수를 따로 갖지 않는다.

## API

| Method | URL | 역할 |
| --- | --- | --- |
| POST | `/predict` | `{"sequence": [{"energy_kwh", "humi_pct", "temp_F"} × SEQ_LEN]}` → `{"predicted_energy_kwh", "model_version"}` |
| POST | `/predict/batch-test` | `{"energy_series": [SEQ_LEN + N개]}` → N건 예측 + 드리프트 판정 |
| GET | `/health` | 서버·모델 준비 상태 |
| POST | `/data/upload` | CSV 업로드 (필수 컬럼 위 표, 측정값 완전 행 SEQ_LEN + WINDOW_SIZE 이상) |
| GET | `/data/status` | 최신 업로드 요약 |
| GET | `/logs`, `/logs/{파일명}` | 로그 파일 조회 |
| GET | `/system/info` | 설정값 (입력 길이, 피처, 게이트, 판정 윈도우, 임계값, epoch, 학습률, 환경 변수) |
| GET | `/system/timeseries` | 예측 요청 로그 구간별 집계 (`window_sec`, `buckets`) |
| GET | `/system/alerts` | `logs/aiops.log`의 [WARN]/[INFO]/[OK] 기록 |
| GET | `/system/dataset` | 최신 업로드 CSV 통계 |
| GET | `/models/current` | 현재 운영 모델 (가장 최근 Production 버전) |
| GET | `/models/history` | MLflow 등록 버전 이력 |
| GET | `/metrics/summary` | 예측 요청 전체 요약 (요청 수, 평균 응답시간, 오류율) |

`/predict` 경로 요청은 `serving_app/monitoring/logger.py` 미들웨어가 `logs/requests.log`에 기록한다.

## 진행 상황

### 완료: 백엔드 · 프론트엔드

- `serving_app/schemas.py`: 요청·응답 필드를 CSV 컬럼명으로 변경 (`HourlyPoint`, `predicted_energy_kwh`, `energy_series`)
- `serving_app/routers/predict.py`, `data.py`: 새 필드명, 업로드 검증, 판정 윈도우 상수 사용
- `serving_app/routers/system.py`, `models.py`, `metrics.py`, `serving_app/monitoring/logger.py`: 대시보드용 조회 API 추가
- `data/storage.py`: 대시보드 통계·업로드 검증용 CSV 읽기 함수(`read_complete_rows`) 추가
- `serving_app/static/index.html`: 대시보드 전체 교체

### 남은 작업: 모델 (완료 — feat/model-torch-port)

모델 코드를 PyTorch 에너지 기준으로 포팅했다 (`Close`/`Volume` 잔재 제거됨).

1. `data/features.py` — 에너지 lag 규격 (`load_splits`, `FEATURE_COLS`, `TARGET_COL`,
   `TIME_COL`, `EnergyScaler`, `tensors_from_rows`), `SEQ_LEN = 24`, 3피처 스케일러
2. `serving_app/lstm_model.py` — `N_FEATURES = 3`, PyTorch `LSTMRegressor`
3. `serving_app/model_loader.py` — `predict_one(sequence)` 계약:
   - 입력: `[{"energy_relative_pct": float, "humi_pct": float, "temp_F": float}, ...]` (SEQ_LEN개)
   - 출력: 변화율 raw 스칼라 (kWh 역변환은 서빙팀)
   - `MLFLOW_MODEL_URI` = `models:/GIGA_Energy_LSTM/Production` (`MODEL_NAME`과 일치)
4. `serving_app/train_and_register.py`
   - `MODEL_NAME = "GIGA_Energy_LSTM"`, epoch 상수 유지
   - `RMSE_GATE = None` (변화율 기준 게이트 미확정 — 데이터 확정 후 팀 합의)
5. `serving_app/monitoring/drift_detector.py`
   - `RMSE_THRESHOLD = None` (게이트와 함께 확정 예정), `WINDOW_SIZE = 21` 유지
6. `serving_app/monitoring/retrain_trigger.py`
   - `WINDOW_SIZE` 상수 사용, 로그 모델명 `MODEL_NAME` 연동
   - 최근 행 조회는 `read_complete_rows` (에너지 컬럼) 경유
7. `scripts/simulate_drift.py`
   - `energy_series` 요청 필드 반영, `BATCH_N = SEQ_LEN + WINDOW_SIZE`
   - 빌드용 시드 CSV 교체(`sample_haic_prices.csv` → 전력 데이터)는 데이터셋 담당과 협의 필요
   - baseline 스크립트(`train_baseline_v1.py`)는 삭제 — baseline `.pt`는 노트북 산출물
