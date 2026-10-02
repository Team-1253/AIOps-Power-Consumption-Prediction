"""
드리프트 감지 시뮬레이션.

핵심 프로세스:
    1) 실제 데이터 조회 - 서버의 GET /data/sample 로 최근 업로드 CSV의 마지막 BATCH_N시간 실제 측정값을 받음
                          (드리프트 배치는 서버가 같은 구간의 사용량에 노이즈를 섞어 돌려줌)
    2) 정상 입력 테스트 - 실제 측정값으로 예측 -> RMSE가 임계값(RMSE_THRESHOLD) 이내인지 확인 (베이스라인)
    3) 드리프트 데이터 주입 - 노이즈를 섞은 배치를 서빙 서버에 전송
    4) 결과 관찰       - RMSE 상승 -> 알림 로그 발생 -> 재학습 트리거 확인

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

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving_app.monitoring.drift_detector import WINDOW_SIZE
from serving_app.schemas import INPUT_LEN

# 보낼 서버 목록. API_URL 은 main() 에서 --target 에 따라 바뀝니다.
TARGETS = {
    "local": "http://localhost:8077",      # 로컬 uvicorn 서버
    "container": "http://localhost:8099",  # 도커 컨테이너
}
API_URL = f"{TARGETS['local']}/predict/batch-test"

# INPUT_LEN(25) + WINDOW_SIZE(24) = 49개를 보내야 배치 하나당 정확히 WINDOW_SIZE(24)개의
# (predicted, actual) 쌍이 쌓여, drift_detector.py가 바로 판정할 수 있다.
BATCH_N = INPUT_LEN + WINDOW_SIZE


def fetch_batch(server: str, drift: bool, n=BATCH_N) -> list[dict]:
    """1단계: 서버의 최근 업로드 CSV에서 마지막 n시간 실제 측정값을 받는다 (drift=True 면 사용량에 노이즈)."""
    resp = requests.get(f"{server}/data/sample", params={"rows": n, "drift": drift})
    resp.raise_for_status()
    return resp.json()["sequence"]


def send_batch(sequence: list[dict], label: str) -> dict:
    """배치를 /predict/batch-test 엔드포인트로 전송한다."""
    resp = requests.post(API_URL, json={"sequence": sequence})
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

    # 배치는 첫 번째 서버에서 한 번만 받는다: 두 서버에 "똑같은" 입력을 보내야 결과를 공정하게 비교할 수 있다.
    source = TARGETS[targets[0]]
    try:
        normal_batch = fetch_batch(source, drift=False)
        drift_batch = fetch_batch(source, drift=True)
    except requests.exceptions.ConnectionError:
        print(f"[stop] {source} 에 연결할 수 없습니다. 서버가 떠 있는지(/health) 확인하세요.")
        return
    except requests.exceptions.HTTPError as e:
        print(f"[stop] 실제 데이터 조회 실패: {e.response.text} (대시보드에서 CSV를 먼저 업로드하세요)")
        return
    print(f"[1] 실제 데이터 {len(normal_batch)}시간 조회 ({source}/data/sample)")

    results = {}
    for name in targets:
        API_URL = f"{TARGETS[name]}/predict/batch-test"
        print(f"\n=== {name} ({TARGETS[name]}) ===")
        try:
            print("[2] 정상 입력 테스트 전송...")
            normal = send_batch(normal_batch, label=f"{name}/normal")
            print("[3] 드리프트 입력 주입...")
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

    print("\n[4] 결과 확인: 각 대시보드의 재학습 로그(" + ", ".join(f"{TARGETS[n]}/" for n in targets)
          + ") 또는 서버 콘솔에서 [WARN] drift detected 로그를 확인하세요.")


if __name__ == "__main__":
    main()
