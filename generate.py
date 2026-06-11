"""
generate.py — Stage 5 of the RAG pipeline: Generation (grounded).

ask(question) retrieves the top-k chunks, builds a numbered context block,
and asks Groq's llama-3.3-70b-versatile to answer using ONLY that context.
The returned sources are extracted programmatically from the retrieval
metadata — never parsed from the LLM's text — so they can't be hallucinated.

Run with:  python generate.py
Requires:  GROQ_API_KEY in .env  (loaded via python-dotenv)
"""

import os

from dotenv import load_dotenv
from groq import Groq

from retrieve import retrieve  # reuse the existing semantic-search function

# Load .env so GROQ_API_KEY is available; never hardcode the key.
load_dotenv()

MODEL = "llama-3.3-70b-versatile"

# The exact system prompt, used verbatim — strict grounding, fixed fallback.
SYSTEM_PROMPT = (
    "You are a career advisor. Answer the user's question using ONLY the "
    "information in the provided documents below. Do not use any outside "
    "knowledge. If the documents do not contain enough information to answer "
    "the question, respond with exactly: 'I don't have enough information "
    "on that in my sources.' Always end your answer with a Sources line "
    "listing the document names you used."
)

# One client for the module. Reads GROQ_API_KEY from the environment.
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _format_context(chunks):
    """Turn retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        # Each entry: an index, the source name, then the chunk text.
        blocks.append(f"[{i}] source: {c['source']}\ntext: {c['text']}")
    return "\n\n".join(blocks)


def ask(question, k=5):
    """Answer `question` grounded ONLY in the top-k retrieved chunks.

    Returns {"answer": <llm text>, "sources": [unique source names]}.
    `sources` is built from retrieve()'s metadata, NOT the LLM output.
    """
    # 1) Retrieve the most relevant chunks for this question.
    chunks = retrieve(question, k=k)

    # 2) Build the numbered context block the model must rely on.
    context = _format_context(chunks)

    # 3) Ask Groq. The user message carries the documents + the question;
    #    temperature=0 keeps the answer deterministic and on-context.
    user_message = (
        f"Documents:\n{context}\n\n"
        f"Question: {question}"
    )
    response = _client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = response.choices[0].message.content

    # 5) Extract unique sources from the RETRIEVED chunks (not the LLM text),
    #    preserving retrieval order. dict.fromkeys de-dupes while keeping order.
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    # Quick manual grounding check across on-topic and off-topic questions.
    questions = [
        "What does a UX researcher do day to day?",
        "What non-SWE careers do CS students talk about switching to?",
        "What is the best restaurant in New York?",  # off-topic -> fallback
    ]
    for q in questions:
        print("\n" + "=" * 80)
        print(f"Q: {q}")
        result = ask(q)
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES (from metadata): {result['sources']}")
