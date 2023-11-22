# /tcdata/trainning_data.pdf
MAX_SENTENCE_LEN=18
THRESHOLD=-130
TEMPERATURE=0.8
TOP_P=0.7

INPUT_PDF_PATH="data/trainning_data.pdf"
# pdf2htmlEX --embed cfijo --dest-dir pdf_output $INPUT_PDF_PATH
# html2text pdf_output/trainning_data.html utf-8 --ignore-links --escape-all > data/raw.txt

# run file for python
python -W ignore retrieve_info.py --local_run --embedding_model stella --threshold $THRESHOLD --max_sentence_len $MAX_SENTENCE_LEN
CUDA_VISIBLE_DEVICES=0 python query_glm.py --local_run --temperature $TEMPERATURE --top_p $TOP_P 
# CUDA_VISIBLE_DEVICES=1 python query_bc.py --local_run
# CUDA_VISIBLE_DEVICES=2 python query_qw.py --local_run --temperature $TEMPERATURE --top_p $TOP_P
python src/generator.py