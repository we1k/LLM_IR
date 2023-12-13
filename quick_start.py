from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# model_path = './tcdata/qwen/Qwen-14B-Chat-Int4'
model_path = "TheBloke/Qwen-14B-Chat-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda().eval()

while True:
    inputs = input("")
    response, history = model.chat(tokenizer, inputs, history=[], max_length=1024, top_p=0.9, temperature=0.9)
    print(response)

