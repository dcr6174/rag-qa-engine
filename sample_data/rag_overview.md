# Retrieval-Augmented Generation Overview

Retrieval-augmented generation, or RAG, is an architecture that combines a
search system with a language model. Instead of relying only on what a model
memorized during training, RAG retrieves relevant passages from a document
collection at question time and asks the model to answer using those passages.

A RAG system has three stages. The ingestion stage splits documents into
chunks and turns each chunk into an embedding vector. The retrieval stage
embeds the question and finds the closest chunks by cosine similarity. The
generation stage composes an answer from the retrieved chunks and cites the
sources it used.

RAG reduces hallucination because answers are grounded in retrieved text. It
also keeps knowledge current: updating the document collection changes the
answers without retraining any model. Every claim in the answer can be traced
back to a source passage, which makes the system auditable.
