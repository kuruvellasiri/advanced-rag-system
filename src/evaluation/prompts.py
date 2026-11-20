"""Prompt templates for optional Gemini-based answer evaluation."""

EVALUATION_SYSTEM_PROMPT = """You are a strict evaluation judge for a RAG system.
Score the generated answer against the question, reference answer, and retrieved
context. Return ONLY valid JSON with numeric scores from 0.0 to 1.0.

Definitions:
- answer_relevance: how directly and completely the answer addresses the question.
- faithfulness: whether every factual claim in the answer is supported by the retrieved context.
- context_relevance: how relevant the retrieved context is to answering the question.
- correctness: how well the answer agrees with the reference answer.

Do not reward unsupported outside knowledge. If the answer says the information
is unavailable, score correctness only against whether the reference answer also
says it is unavailable."""

EVALUATION_HUMAN_PROMPT = """Question:
{question}

Reference answer:
{reference_answer}

Retrieved context:
{context}

Generated answer:
{answer}

Return JSON with exactly these keys:
{{"answer_relevance": 0.0, "faithfulness": 0.0, "context_relevance": 0.0, "correctness": 0.0, "reasoning": "short explanation"}}"""


def build_evaluation_prompt(
    *,
    question: str,
    reference_answer: str,
    context: str,
    answer: str,
) -> list:
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", EVALUATION_SYSTEM_PROMPT),
        ("human", EVALUATION_HUMAN_PROMPT),
    ])
    return prompt.format_messages(
        question=question,
        reference_answer=reference_answer,
        context=context,
        answer=answer,
    )
