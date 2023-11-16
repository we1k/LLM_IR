# /tcdata/trainning_data.pdf
MAX_SENTENCE_LEN=29

INPUT_PDF_PATH="/tcdata/trainning_data.pdf"

pdf2htmlEX --embed cfijo --dest-dir pdf_output $INPUT_PDF_PATH
html2text pdf_output/trainning_data.html utf-8 --ignore-links --escape-all > data/raw.txt


# run file for python
python retrieve_info.py --embedding_model stella
python query_glm.py
python query_bc.py
python query_qw.py
python src/generator.py
cp result/submit.json /app/result.json