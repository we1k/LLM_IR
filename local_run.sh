echo "Good Luck!"

# /tcdata/trainning_data.pdf
MAX_SENTENCE_LEN=18
THRESHOLD=-140
TEMPERATURE=0.8
TOP_P=0.95
SEED=42

INPUT_PDF_PATH="data/trainning_data.pdf"
embedding_model="stella"
###################
# section related_str
# pdf2htmlEX --embed cfijo --dest-dir pdf_output $INPUT_PDF_PATH
# html2text pdf_output/trainning_data.html utf-8 --ignore-links --escape-all > data/raw.txt
echo "Starting Retrieval"
# python -W ignore retrieve_info.py --local_run --embedding_model $embedding_model --threshold $THRESHOLD --max_sentence_len $MAX_SENTENCE_LEN

echo "Starting Reranking"
# python re-ranker.py

echo "Starting Generation"
CUDA_VISIBLE_DEVICES=2 python3 query_qw_flash_attn.py --local_run --temperature 0.7 --top_p 0.95 --seed 1203 --use-14B --output chatglm 
CUDA_VISIBLE_DEVICES=2 python3 query_qw_flash_attn.py --local_run --temperature $TEMPERATURE --top_p $TOP_P --seed $SEED --output baichuan --beam_search --best_of 3
CUDA_VISIBLE_DEVICES=2 python3 query_qw_flash_attn.py --local_run --temperature 0.4 --top_p 0.95 --seed 521 --use-1_8B --output qianwen --prompt_idx 2

python src/generator.py