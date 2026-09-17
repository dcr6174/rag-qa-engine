# Evaluating RAG Systems

A RAG system is evaluated on two axes: retrieval quality and answer quality.

Retrieval quality asks whether the right passages were found. Hit rate at k
measures how often at least one relevant passage appears in the top k
results. Mean reciprocal rank, or MRR, also considers the position of the
first relevant passage, rewarding systems that rank it first.

Answer quality asks whether the response is faithful to the retrieved
passages and relevant to the question. Faithfulness means every claim in the
answer is supported by a retrieved source. A simple lexical proxy compares
the content words of each answer sentence against the retrieved passages;
stronger setups use a second model as a judge.

Evaluation requires a labelled dataset of questions with their expected
source documents. Small, curated sets of twenty to fifty questions already
catch regressions when chunking, embeddings, or prompts change.
