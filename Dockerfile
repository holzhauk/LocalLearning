# syntax=docker/dockerfile:1
FROM python:3.9-slim-buster
RUN mkdir -p /pytorch-dev \
    && mkdir -p /pytorch-dev/src \
    && mkdir -p /pytorch-dev/notebooks
WORKDIR /pytorch-dev
COPY ./requirements.txt requirements.txt
EXPOSE 8080
EXPOSE 3108
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    torch==1.10.2 \
    torchvision==0.11.3 \
    torchaudio==0.10.2
RUN groupadd --gid 1000 pytorch-cpu \
    && useradd --uid 1000 --gid 1000 -m pytorch-cpu
USER pytorch-cpu
CMD ["bash"]
