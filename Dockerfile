# w/ flash-attn
FROM registry.cn-shanghai.aliyuncs.com/we1k/irs:env
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  transformers_stream_generator rank_bm25
RUN apt-get -y install curl

COPY . /app
WORKDIR /app

RUN mkdir -p /app/models
RUN mv /app/stella-base-zh-v2 /app/models/stella-base-zh-v2

CMD ["bash", "run.sh"]


# w/o attn
# FROM registry.cn-hangzhou.aliyuncs.com/we1k/smp_env:v1
# COPY . /app
# WORKDIR /app
# CMD ["bash", "run.sh"]