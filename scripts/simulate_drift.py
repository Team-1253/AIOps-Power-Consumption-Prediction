"""
Day3 드리프트 감지 시뮬레이션 (119~123번 슬라이드).

핵심 프로세스:
    1) 기준 통계 산출   - 학습에 쓴 3년치 HAIC 데이터의 평균·표준편차 계산
    2) 정상 입력 테스트 - 같은 분포의 데이터로 예측 -> RMSE $4 이내 확인 (베이스라인)
    3) 드리프트 데이터 생성 - 변동성을 인위적으로 3배 키운 가격 데이터 생성
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
from data.features import load_rows
from data.storage import latest_upload

# 보낼 서버 목록. API_URL 은 main() 에서 --target 에 따라 바뀝니다.
TARGETS = {
    "local": "http://localhost:8077",      # 로컬 uvicorn 서버
    "container": "http://localhost:8099",  # 도커 컨테이너
}
API_URL = f"{TARGETS['local']}/predict/batch-test"

# 기준 통계용 데이터: 호스트의 data/uploads/ 에 업로드한 CSV가 없으면 학습에 쓰인 예시 데이터로 계산합니다.
SAMPLE_CSV = "data/sample_haic_prices.csv"


def compute_baseline_stats(csv_path: str | None = None) -> tuple[float, float]:
    """1단계: 학습에 사용한 데이터(업로드된 최신 CSV)의 평균·표준편차."""
    if csv_path is None:
        try:
            csv_path = latest_upload()
        except FileNotFoundError:
            csv_path = SAMPLE_CSV
            print(f"[info] 업로드된 CSV가 없어 {SAMPLE_CSV} 로 기준 통계를 계산합니다.")
    rows = load_rows(csv_path)
    closes = np.array([r["Close"] for r in rows])
    return float(closes.mean()), float(closes.std())


# SEQ_LEN(20) + WINDOW_SIZE(21) = 41개를 보내야 배치 하나당 정확히 WINDOW_SIZE(21)개의
# (predicted, actual) 쌍이 쌓여, drift_detector.py가 바로 판정할 수 있다.
BATCH_N = 41

# 학습 데이터(실제 IBM 시세 기반)는 추세·모멘텀이 있는 시계열이라, 평균 주변의 순수
# 백색잡음(iid noise)을 넣으면 "정상" 입력조차 모델이 못 맞춰 오탐(false positive)이
# 납니다. 그래서 정상/드리프트 배치 모두 일별 수익률(log return) 기반의 랜덤워크로
# 만들고, 그 수익률의 표준편차(변동성)만 다르게 줍니다.
NORMAL_SIGMA = 0.012  # 학습 데이터의 안정적 구간과 비슷한 일별 변동성 (~1.2%)
DRIFT_SIGMA = NORMAL_SIGMA * 3  # 변동성을 3배 키운 드리프트


def _random_walk(n: int, base: float, sigma: float) -> np.ndarray:
    log_returns = np.random.normal(0, sigma, n)
    return base * np.exp(np.cumsum(log_returns))


def generate_normal_batch(n=BATCH_N, base=165.0, sigma=NORMAL_SIGMA):
    """학습 데이터와 비슷한 변동성의 정상 입력(랜덤워크)."""
    return _random_walk(n, base, sigma)


def generate_drift_batch(n=BATCH_N, base=165.0, sigma=DRIFT_SIGMA):
    """변동성을 3배 키운 드리프트 입력 (의도적으로 오차 유발)."""
    return _random_walk(n, base, sigma)


def send_batch(prices: np.ndarray, label: str) -> dict:
    """Day3 정답 구현: 생성한 배치를 /predict/batch-test 엔드포인트로 전송한다."""
    resp = requests.post(API_URL, json={"prices": prices.tolist()})
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
    parser = argparse.ArgumentParser(description="HAIC 드리프트 감지 시뮬레이션")
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
