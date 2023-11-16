# FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04
FROM registry.cn-hangzhou.aliyuncs.com/we1k/smp_env:v0

RUN pip install deepspeed html2text transformers==4.34.1 langchain==0.0.312

COPY . /app
WORKDIR /app

CMD ["bash", "run.sh"]