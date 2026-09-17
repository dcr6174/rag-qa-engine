# Testing Strategy for AI Pipelines

AI pipelines need the same testing discipline as any production service.

Unit tests cover deterministic components: the chunker respects size limits
and keeps overlap, the embedder returns normalized vectors, and the store
returns results sorted by similarity. These tests run offline and must stay
fast, so a deterministic hashing embedder stands in for neural models.

Integration tests run the pipeline end to end on a fixed corpus. They assert
that a question about a topic retrieves that topic's document first, and that
the answer cites its sources. Golden datasets freeze expected behavior so a
change in retrieval ranking fails loudly.

Regression evaluation runs the labelled QA set after every change to
chunking, embedding, or generation settings. Metric drops are treated like
failing tests: the change does not ship until the numbers recover.
