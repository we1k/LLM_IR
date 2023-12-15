# w/ flash-attn
# FROM registry.cn-shanghai.aliyuncs.com/we1k/irs:env_2
FROM registry.cn-shanghai.aliyuncs.com/we1k/irs:env_3

COPY . /app
WORKDIR /app


CMD ["bash", "run.sh"]


# w/o attn
# FROM registry.cn-hangzhou.aliyuncs.com/we1k/smp_env:v1
# COPY . /app
# WORKDIR /app
# CMD ["bash", "run.sh"]