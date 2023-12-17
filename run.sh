echo "Good Luck!"

# /tcdata/trainning_data.pdf
MAX_SENTENCE_LEN=18
THRESHOLD=-140
TEMPERATURE=0.8
TOP_P=0.95
SEED=520
PROMPT="根据已有信息与问题最相关的部分，完整的回答问题。"


INPUT_PDF_PATH="/tcdata/trainning_data.pdf"
embedding_model="stella"

################### preprocess
echo "Starting Preprocess"
pdf2htmlEX --embed cfijo --dest-dir pdf_output $INPUT_PDF_PATH > /dev/null
html2text pdf_output/trainning_data.html utf-8 --ignore-links --escape-all > data/raw.txt

echo "Starting Retrieval"
python3 -W ignore retrieve_info.py --embedding_model $embedding_model --threshold $THRESHOLD --max_sentence_len $MAX_SENTENCE_LEN

echo "Starting Reranking"
python3 re-ranker.py

echo "Starting Generation"
# run generation
# python3 query_qw.py --temperature $TEMPERATURE --top_p $TOP_P  --seed $SEED
python3 query_qw_flash_attn.py --temperature 0.7 --top_p 0.95 --seed 1203 --use-14B --output chatglm 
python3 query_qw_flash_attn.py --temperature $TEMPERATURE --top_p $TOP_P --seed $SEED --output baichuan --beam_search --best_of 3
python3 query_qw_flash_attn.py --temperature 0.4 --top_p 0.95 --seed 521 --use-1_8B --output qianwen --prompt_idx 2
# python3 query_qw_flash_attn.py --temperature $TEMPERATURE --top_p $TOP_P --seed $SEED --use-14B --output chatglm
# python3 query_qw_flash_attn.py --temperature $TEMPERATURE --top_p $TOP_P --seed $SEED --output baichuan --prompt_idx 2
# python3 query_qw_flash_attn.py --temperature 0.4 --top_p 0.95 --seed 521 --use-1_8B --output qianwen --prompt_idx 3

python3 src/generator.py
cp result/submit.json /app/result.json