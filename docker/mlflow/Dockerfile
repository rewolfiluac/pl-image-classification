FROM python:3.7

ARG UID
ARG GID

RUN pip install --upgrade pip && \
    pip install mlflow psycopg2 boto3

# sudo権限を持つ一般ユーザーを作成
ENV USER mlflow
ENV GROUP mlflow
ENV HOME /home/${USER}
ENV SHELL /bin/bash

RUN groupadd -g ${GID} ${GROUP}
RUN useradd -u ${UID} -g ${GROUP} -m ${USER}
RUN gpasswd -a ${USER} sudo
RUN echo "${USER}:mlflow" | chpasswd
RUN sed -i.bak "s#${HOME}:#${HOME}:${SHELL}#" /etc/passwd

USER ${USER}
WORKDIR ${HOME}
