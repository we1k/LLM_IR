echo "Good Luck!"

# /tcdata/trainning_data.pdf
MAX_SENTENCE_LEN=18
THRESHOLD=-140
TEMPERATURE=0.7
TOP_P=0.8
SEED=1203

INPUT_PDF_PATH="/tcdata/trainning_data.pdf"
embedding_model="stella"

################### preprocess
pdf2htmlEX --embed cfijo --dest-dir pdf_output $INPUT_PDF_PATH
html2text pdf_output/trainning_data.html utf-8 --ignore-links --escape-all > data/raw.txt

echo "Starting Retrieval"
python -W ignore retrieve_info.py --embedding_model $embedding_model --threshold $THRESHOLD --max_sentence_len $MAX_SENTENCE_LEN

echo "Starting Reranking"

python re-ranker.py

echo "Starting Generation"
# run generation
python query_qw.py --temperature $TEMPERATURE --top_p $TOP_P  --seed $SEED
python src/generator.py
cp result/submit.json /app/result.json