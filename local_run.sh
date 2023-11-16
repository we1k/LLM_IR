# /tcdata/trainning_data.pdf
MAX_SENTENCE_LEN=29

INPUT_PDF_PATH="data/trainning_data.pdf"
pdf2htmlEX --embed cfijo --dest-dir pdf_output $INPUT_PDF_PATH
html2text pdf_output/trainning_data.html utf-8 --ignore-links --escape-all > data/raw.txt

# run file for python
python retrieve_info.py --local_run --embedding_model stella --max_sentence_len $MAX_SENTENCE_LEN
python query_glm.py --local_run
python query_bc.py --local_run
python query_qw.py --local_run
python src/generator.py