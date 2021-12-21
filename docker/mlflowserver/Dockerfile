ARG BASE_CONTAINER=python:3.7
FROM ${BASE_CONTAINER}

ENV LANG C.UTF-8
WORKDIR /mlflowserver
COPY requirements.txt /mlflowserver/requirements.txt
RUN pip3 install -U pip
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY servers/mlflowserver/mlflowserver/before-run /app/before-run
COPY servers/mlflowserver/mlflowserver/image_metadata.json /app/image_metadata.json
COPY servers/mlflowserver/mlflowserver/MLFlowServer.py /app/MLFlowServer.py
COPY servers/mlflowserver/mlflowserver/pip_env_create.py /app/pip_env_create.py
COPY servers/mlflowserver/mlflowserver/requirements.txt /app/requirements.txt
COPY servers/mlflowserver/mlflowserver/serving_utils.py /app/serving_utils.py
RUN chmod +x /app/before-run
RUN chmod +x /app/pip_env_create.py
RUN chmod +r /app/MLFlowServer.py

EXPOSE 5000
EXPOSE 9000

ENV MODEL_NAME MLFlowServer
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

ENV PYTHONPATH /app:${PYTHONPATH}

CMD bash before-run && \
    exec seldon-core-microservice ${MODEL_NAME} --service-type ${SERVICE_TYPE} --persistence ${PERSISTENCE}
