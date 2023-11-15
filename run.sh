# builder for pdf2html
sudo apt install ./builder/pdf2htmlEX.deb
pip install html2text

# pdf2html
pdf2htmlEX --embed cfijo --dest-dir pdf_output data/QA.pdf

# html2text
html2text pdf_output/QA.html utf-8 --ignore-links --escape-all > data/raw.txt

# run file for python
python retrieve_info.py --embedding_model stella