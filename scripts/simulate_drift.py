"""
Day3 드리프트 감지 시뮬레이션 (전력 사용량 에너지판).

핵심 프로세스:
    1) 기준 통계 산출   - 업로드된 최신 전력 CSV의 energy_kwh 평균·표준편차 계산
    2) 정상 입력 테스트 - 같은 분포의 데이터로 예측 -> RMSE 기준 이내 확인 (베이스라인)
    3) 드리프트 데이터 생성 - 변동성을 인위적으로 3배 키운 사용량 데이터 생성
    4) 드리프트 데이터 주입 - 생성한 데이터를 서빙 서버에 연속 요청으로 전송
    5) 결과 관찰       - RMSE 상승 -> 알림 로그 발생 -> 재학습 트리거 확인

사전 준비: 서빙 서버가 이미 떠 있어야 합니다. 이 스크립트는 서버 "밖"(호스트 터미널, 프로젝트 루트)에서
          실행하는 외부 클라이언트입니다.

실행 (--target 으로 보낼 서버 선택)
    python scripts/simulate_drift.py                     # 로컬 uvicorn 서버 (8077, 기본값)
    python scripts/simulate_drift.py --target container  # 도커 컨테이너 (8099)
    python scripts/simulate_drift.py --target both       # 같은 배치를 두 서버에 보내 결과 비교

    두 서버 동시 기동 예)
      터미널 1: MODEL_SOURCE=mlflow uvicorn serving_app.main:app --host 0.0.0.0 --port 8077
      터미널 2: docker compose -f serving_app/docker-compose.yml up --build
"""
import argparse
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.features import SEQ_LEN
from data.storage import latest_upload, read_complete_rows
from serving_app.monitoring.drift_detector import WINDOW_SIZE

# 보낼 서버 목록. API_URL 은 main() 에서 --target 에 따라 바뀝니다.
TARGETS = {
    "local": "http://localhost:8077",      # 로컬 uvicorn 서버
    "container": "http://localhost:8099",  # 도커 컨테이너
}
API_URL = f"{TARGETS['local']}/predict/batch-test"

# 기준 통계용 데이터: 호스트의 data/uploads/ 에 업로드한 CSV가 없으면 정제 예시 데이터로 계산합니다.
SAMPLE_CSV = "data/hourly_clean.csv"
ENERGY_COL = "energy_kwh"


def compute_baseline_stats(csv_path: str | None = None) -> tuple[float, float]:
    """1단계: 사용 데이터(업로드된 최신 CSV)의 energy_kwh 평균·표준편차."""
    if csv_path is None:
        try:
            csv_path = latest_upload()
        except FileNotFoundError:
            csv_path = SAMPLE_CSV
            print(f"[info] 업로드된 CSV가 없어 {SAMPLE_CSV} 로 기준 통계를 계산합니다.")
    total, rows = read_complete_rows(csv_path, ("date_utc", "time_utc"), (ENERGY_COL,))
    values = np.array([r[ENERGY_COL] for r in rows])
    print(f"[info] {csv_path}: 전체 {total}행 중 측정값 완비 {len(rows)}행 사용")
    return float(values.mean()), float(values.std())


# SEQ_LEN + WINDOW_SIZE 개를 보내야 배치 하나당 정확히 WINDOW_SIZE개의
# (predicted, actual) 쌍이 쌓여, drift_detector.py가 바로 판정할 수 있다.
BATCH_N = SEQ_LEN + WINDOW_SIZE

# 사용량 시계열은 추세·일주기 패턴이 있어, 평균 주변의 순수 백색잡음(iid noise)을 넣으면
# "정상" 입력조차 모델이 못 맞춰 오탐(false positive)이 납니다. 그래서 정상/드리프트
# 배치 모두 시간별 변화율 기반의 랜덤워크로 만들고, 그 변화율의 표준편차(변동성)만
# 다르게 줍니다.
NORMAL_SIGMA = 0.012  # 안정적 구간의 시간별 변동성 (~1.2%)
DRIFT_SIGMA = NORMAL_SIGMA * 3  # 변동성을 3배 키운 드리프트


def _random_walk(n: int, base: float, sigma: float) -> np.ndarray:
    rel_changes = np.random.normal(0, sigma, n)
    return base * np.exp(np.cumsum(rel_changes))


def generate_normal_batch(n=BATCH_N, base=2500.0, sigma=NORMAL_SIGMA):
    """기준 통계와 비슷한 변동성의 정상 입력(랜덤워크, kWh)."""
    return _random_walk(n, base, sigma)


def generate_drift_batch(n=BATCH_N, base=2500.0, sigma=DRIFT_SIGMA):
    """변동성을 3배 키운 드리프트 입력 (의도적으로 오차 유발)."""
    return _random_walk(n, base, sigma)


def send_batch(series: np.ndarray, label: str) -> dict:
    """생성한 배치를 /predict/batch-test 엔드포인트로 전송한다."""
    resp = requests.post(API_URL, json={"energy_series": series.tolist()})
    resp.raise_for_status()
    result = resp.json()
    print(f"[{label}] drift_check = {result['drift_check']}")
    return result


def _summary(check: dict) -> str:
    if check.get("status") != "retrain_triggered":
        return check.get("status", "?")
    return f"retrain_triggered (promoted={check.get('promoted')}, rmse={check.get('rmse', 0):.2f})"


def main():
    global API_URL
    parser = argparse.ArgumentParser(description="전력 사용량 드리프트 감지 시뮬레이션")
    parser.add_argument("--target", choices=["local", "container", "both"], default="local",
                        help="local=8077, container=8099, both=같은 배치를 두 서버에 보내 비교")
    args = parser.parse_args()
    targets = ["local", "container"] if args.target == "both" else [args.target]

    mean, std = compute_baseline_stats()
    print(f"[1] 기준 통계: mean={mean:.2f}, std={std:.2f}")

    # 배치는 한 번만 만든다: 두 서버에 "똑같은" 입력을 보내야 결과를 공정하게 비교할 수 있다.
    normal_batch = generate_normal_batch(base=mean)
    drift_batch = generate_drift_batch(base=mean)

    results = {}
    for name in targets:
        API_URL = f"{TARGETS[name]}/predict/batch-test"
        print(f"\n=== {name} ({TARGETS[name]}) ===")
        try:
            print("[2] 정상 입력 테스트 전송...")
            normal = send_batch(normal_batch, label=f"{name}/normal")
            print("[3-4] 드리프트 입력 주입...")
            drift = send_batch(drift_batch, label=f"{name}/drift_injection")
            results[name] = (normal["drift_check"], drift["drift_check"])
        except requests.exceptions.ConnectionError:
            print(f"[skip] {TARGETS[name]} 에 연결할 수 없습니다. 서버가 떠 있는지(/health) 확인하세요.")
            results[name] = None

    if len(targets) == 2:
        print("\n[비교] 같은 배치 → 서버별 결과")
        for name in targets:
            r = results[name]
            line = "연결 실패" if r is None else f"normal={_summary(r[0])} | drift={_summary(r[1])}"
            print(f"  {name:<9}: {line}")

    print("\n[5] 결과 확인: 각 대시보드의 재학습 로그(" + ", ".join(f"{TARGETS[n]}/" for n in targets)
          + ") 또는 서버 콘솔에서 [WARN] drift detected 로그를 확인하세요.")


if __name__ == "__main__":
    main()
