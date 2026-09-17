from rag_qa.generate import OfflineGenerator, _UNTRUSTED_SYSTEM_PROMPT, rewrite_query


def test_followup_questions_are_rewritten_standalone():
    history = [{"question": "How does RAG reduce hallucination?"}]
    rewritten = rewrite_query("what about the second one?", history)
    assert rewritten != "what about the second one?"
    assert "hallucination" in rewritten


def test_standalone_questions_pass_through():
    history = [{"question": "How does RAG reduce hallucination?"}]
    question = "What metrics does the evaluation harness report for retrieval quality?"
    assert rewrite_query(question, history) == question
    assert rewrite_query(question, None) == question


def test_first_turn_never_rewritten():
    assert rewrite_query("it?", None) == "it?"
    assert rewrite_query("it?", []) == "it?"


def test_untrusted_prompt_frames_retrieved_text_as_data():
    prompt = _UNTRUSTED_SYSTEM_PROMPT.lower()
    assert "untrusted" in prompt
    assert "never instructions" in prompt
    assert "ignore" in prompt
