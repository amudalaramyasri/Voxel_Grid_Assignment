"""
assistant/assistant.py — A lightweight AI assistant that can answer questions
about this project (Part 6 of the assignment).

Three modes, chosen automatically, in priority order:
  1. Ollama (open-source, local, free): if a local Ollama server is running
     (https://ollama.com), questions are answered by a small local model
     (e.g. llama3.2:1b), grounded in PROJECT_FACTS + results/metrics.json
     via the system prompt. No API key, no cost, fully offline.
  2. Anthropic API (optional): if ANTHROPIC_API_KEY is set and Ollama isn't
     running, falls back to Claude.
  3. Rule-based fallback: if neither LLM backend is available, a simple
     keyword router answers from the same PROJECT_FACTS dict, so the
     assistant still works out of the box with zero setup.

This is intentionally lightweight per the assignment brief -- the point is to
show LLM integration into the ML workflow, not to build a production agent.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assistant.knowledge import PROJECT_FACTS, build_context_block

# Change this to whichever model you've pulled with `ollama pull <model>`.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

SYSTEM_PROMPT_TEMPLATE = """You are a friendly, knowledgeable assistant for a \
COVID-19 CT-scan classification ML project. You can answer questions about \
the problem it solves, the dataset, the model architecture and why it was \
chosen, how training works, evaluation results, limitations, how results \
should be interpreted, how the model could be improved, and how it should \
(and shouldn't) be deployed in a real clinical setting.

Ground every answer in the project facts below -- don't invent numbers or \
details that aren't there. If someone asks something this project genuinely \
doesn't cover, say so plainly rather than guessing, but otherwise feel free \
to explain, compare, and reason about the facts naturally (e.g. you can \
combine multiple facts, give examples, or restate things in different ways \
depending on how the question is phrased). Keep answers concise (2-6 \
sentences unless more detail is clearly wanted) and in plain language \
suitable for a radiologist or technologist, not just an ML engineer.

{context}
"""

# Keyword -> fact key, used by the rule-based fallback.
KEYWORD_ROUTES = [
    (("what problem", "what does", "solve", "purpose", "goal"), "problem"),
    (("clinical", "why is it relevant", "why relevant", "useful for"), "clinical_relevance"),
    (("dataset", "data set", "data used", "which data", "how many image",
      "class distribution", "kaggle"), "dataset"),
    (("why was this model", "model selected", "architecture", "why mobilenet",
      "which model", "model selection", "selected", "why this model",
      "backbone"), "model"),
    (("how was", "trained", "training", "hyperparameter", "epoch",
      "batch size", "learning rate", "optimizer"), "training"),
    (("limitation", "weakness", "fail", "fails", "wrong", "problem with",
      "downside"), "limitations"),
    (("interpret", "confidence", "probability", "trust", "result",
      "accuracy", "performance", "how good", "how accurate"), "interpretation"),
    (("improve", "better", "next step", "future", "enhance", "upgrade"), "improvements"),
    (("deploy", "production", "real world", "clinical use", "hospital use",
      "safe to use"), "deployment"),
    (("assistant", "yourself", "how do you work", "ollama", "llm",
      "are you"), "assistant_design"),
]


def _rule_based_answer(question: str) -> str:
    q = question.lower()
    matched = []
    for keywords, fact_key in KEYWORD_ROUTES:
        if any(k in q for k in keywords) and fact_key not in matched:
            matched.append(fact_key)

    if matched:
        return "\n\n".join(PROJECT_FACTS[k] for k in matched)

    topics = ["the problem it solves", "the dataset", "why this model was chosen",
              "how it was trained", "its limitations", "how to interpret results",
              "how it could be improved", "how it should be deployed"]
    return (
        "I don't have a fixed answer for that exact phrasing, but I can tell you "
        "about: " + ", ".join(topics) + ". Try asking about one of those, or for "
        "more natural/free-form answers, run a local Ollama model (see README) "
        "so I can reason over the project facts directly instead of just "
        "keyword-matching."
    )


def _ollama_available() -> bool:
    import requests
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_answer(question: str, history=None) -> str:
    import requests

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=build_context_block())
    messages = [{"role": "system", "content": system_prompt}]
    messages += list(history or [])
    messages.append({"role": "user", "content": question})

    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _anthropic_answer(question: str, history=None) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=build_context_block())

    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=system_prompt,
        messages=messages,
    )
    return "".join(block.text for block in response.content if block.type == "text")


def answer(question: str, history=None) -> str:
    """Public entry point used by the CLI."""
    if _ollama_available():
        try:
            return _ollama_answer(question, history)
        except Exception as exc:
            print(f"[assistant] Ollama call failed ({exc}); trying next backend.")

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _anthropic_answer(question, history)
        except Exception as exc:
            print(f"[assistant] Anthropic call failed ({exc}); falling back to rule-based answer.")

    return _rule_based_answer(question)


def main():
    print("Project assistant ready. Ask about the model, data, training, or limitations.")
    print("(type 'exit' to quit)\n")
    history = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        a = answer(q, history)
        print(f"assistant> {a}\n")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": a})


if __name__ == "__main__":
    main()
