# FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04
# FROM registry.cn-hangzhou.aliyuncs.com/we1k/smp_env:v1
FROM registry.cn-hangzhou.aliyuncs.com/we1k/smp_env:v0

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple deepspeed html2text transformers==4.34.1 langchain==0.0.312 PyPDF2 spacy

RUN python -m spacy download zh_core_web_sm
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rank_bm25

COPY . /app
WORKDIR /app

CMD ["bash", "run.sh"]