index:
	PYTHONPATH=src python src/run_indexer.py

ingestion:
	PYTHONPATH=src python src/run_ingestion.py

sqlite:
	PYTHONPATH=src python src/run_sqlite.py

query:
	PYTHONPATH=src python src/query.py \
	  "Explain incident INC-999999" \
	  --tenant acme \
	  --groups sre \
	  --provider bedrock

ask-rerank:
	RAG_LLM_MODEL_ID="us.amazon.nova-lite-v1:0" \
	PYTHONPATH=src python src/ask.py \
  --tenant acme \
  --groups sre platform \
  --rerank

ask:
	RAG_LLM_MODEL_ID="us.amazon.nova-lite-v1:0" \
	PYTHONPATH=src python src/ask.py \
  --tenant acme \
  --groups sre platform \
  --rerank

eval-manual:
	RAG_LLM_MODEL_ID="us.amazon.nova-lite-v1:0" \
	PYTHONPATH=src python src/evaluate_generation.py \
		--tenant acme \
		--groups sre

eval-manual-rerank:
	RAG_LLM_MODEL_ID="us.amazon.nova-lite-v1:0" \
	PYTHONPATH=src python src/evaluate_generation.py \
		--tenant acme \
		--groups sre \
		--rerank

eval-ret:
	RAG_LLM_MODEL_ID="us.amazon.nova-lite-v1:0" \
	PYTHONPATH=src python src/evaluate_generation.py \
		--tenant acme \
		--groups sre


eval-retrieval:
	PYTHONPATH=src python src/evaluate_retrieval.py \
		--dataset data/evaluation/retrieval.json \
		--region us-east-1

# Evaluate hybrid retrieval with Bedrock reranking
eval-retrieval-rerank:
	PYTHONPATH=src python src/evaluate_retrieval.py \
		--dataset data/evaluation/retrieval.json \
		--region us-east-1 \
		--rerank

verify:
	PYTHONPATH=src python -m compileall -q src
	PYTHONPATH=src pytest -q