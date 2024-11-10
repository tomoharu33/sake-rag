# sake-rag 
JSA Sake Diploma二次試験の設問に対して、適切な回答を行うRAG 
LangChainを用いず、OpenAIのAPIを利用 
see also #sake-rag-chain 

# 参考
https://zenn.dev/spiralai/articles/8af7cbf526c2e1

# 環境構築手順メモ
- pyenv local 3.11.1
- python -m venv .venv
- source .venv/bin/activate
  - 終わったら deactivate
- touch requirements.txt
  - requirementsにpipパッケージ記載
- pip install -r requirements.txt

これにて準備完了
