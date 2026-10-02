# 에너지 LSTM 서빙 앱 컨테이너 이미지 (단일 컨테이너, PyTorch/XPU)
# 빌드/실행은 반드시 프로젝트 루트에서: docker build -f serving_app/Dockerfile -t giga-serving:v1 .
# 참고: torch(XPU runtime)가 들어가서 이미지가 큽니다
#       (수 GB대) - 빌드에 몇 분 걸릴 수 있습니다. (GPU 없는 컨테이너이지만 CPU로 정상 동작합니다)
#
# 포트: 컨테이너는 8099 에서 열립니다 -> 대시보드 http://localhost:8099/
#       (로컬 uvicorn 서버 8077 과 겹치지 않도록 분리)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# fastapi, uvicorn, mlflow, torch, python-multipart(파일 업로드) 포함

COPY serving_app/ serving_app/
COPY data/ data/
COPY scripts/ scripts/

# 대시보드에서 업로드하는 것과 동일한 방식으로, 빌드용 정제 데이터를 data/uploads/에
# 먼저 "업로드"해 둔 뒤 -> Day2 MLflow 학습/등록까지 이미지 빌드 시점에 컨테이너 안에서
# 전부 끝냅니다(로컬에서 미리 실행해둘 필요 없는 자기 완결형 이미지).
# baseline .pt 번들(serving_app/models/energy_lstm.pt)은 repo에 포함되어 COPY로 들어갑니다.
# MLflow도 별도 서버 없이 로컬 sqlite(mlflow.db)로 동작하므로,
# 학습(등록)과 서빙이 항상 같은 컨테이너의 같은 파일시스템(/app) 안에서 이뤄집니다 -
# 여러 컨테이너가 네트워크로 서로의 모델/아티팩트를 찾아가야 하는 구조 자체가 없습니다.
# NOTE: train_and_register.py는 data/train.csv 등 split 파일을 읽으므로,
# 데이터셋 담당이 빌드 컨텍스트에 split CSV를 포함시켜야 Day2 단계가 동작합니다.
RUN mkdir -p data/uploads \
 && cp data/hourly_clean.csv data/uploads/build_seed.csv \
 && python serving_app/train_and_register.py

EXPOSE 8099

ENV MODEL_SOURCE=mlflow
ENV LOADING_MODE=eager

CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8099"]
