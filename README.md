uv add python-docx
uv add beautifulsoup4
uv pip install python-docx
uv pip freeze > requirements.txt
uv pip compile requirements.txt

streamlit run biz_analyzer.py --server.port 8501 --server.address 0.0.0.0
