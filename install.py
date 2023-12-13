from modelscope.hub.snapshot_download import snapshot_download

# model_dir = snapshot_download('ZhipuAI/chatglm3-6b', cache_dir='/home/lz/workspace/Competition/chatglm/chatglm3-6b')

model_dir = snapshot_download('qwen/Qwen-14B-Chat', cache_dir='/home/lzw/project/IR-docker/tcdata')
