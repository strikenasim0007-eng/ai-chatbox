# ai-chatbox
"""
AI Chatbot (ChatGPT-style, English only) - project.py

"""

import os
import re
import sys
import time

# Auto-launcher: lets "python project.py" work without the ScriptRunContext warning
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx() is None:
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())

import requests  # noqa: E402
import streamlit as st  # noqa: E402

SYSTEM_PROMPT = (
    "You are a helpful, accurate and friendly AI assistant, like ChatGPT. "
    "Always answer in clear English. Give step-by-step explanations when useful, "
    "use code blocks for code, and say so honestly if you are not sure."
)

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 AI Chatbot")
st.caption("Ask me anything - answers in English")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("AI engine", ["OpenAI (ChatGPT)", "Anthropic (Claude)", "Ollama (free, local)"])

    api_key, model, ollama_url = "", "", ""
    if provider == "OpenAI (ChatGPT)":
        api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model = st.text_input("Model", value="gpt-4o-mini")
    elif provider == "Anthropic (Claude)":
        api_key = st.text_input("Anthropic API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
        model = st.text_input("Model", value="claude-sonnet-5-5")
    else:
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        model = st.text_input("Model", value="llama3")

    temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.slider("Max answer length (tokens)", 200, 4000, 1000, 100)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------------- AI calls ----------------
def ask_openai(history):
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def ask_anthropic(history):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": SYSTEM_PROMPT,
            "messages": history,
        },
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Anthropic error {r.status_code}: {r.text[:300]}")
    return "".join(b.get("text", "") for b in r.json()["content"])


def ask_ollama(history):
    try:
        r = requests.post(
            f"{ollama_url.rstrip('/')}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=300,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Install it from ollama.com and run: ollama pull llama3")
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
    return r.json()["message"]["content"]


def get_answer(history):
    if provider != "Ollama (free, local)" and not api_key:
        raise RuntimeError("Please paste your API key in the sidebar, or choose Ollama for a free local model.")
    if provider == "OpenAI (ChatGPT)":
        return ask_openai(history)
    if provider == "Anthropic (Claude)":
        return ask_anthropic(history)
    return ask_ollama(history)


def typing_effect(text):
    for chunk in re.findall(r"\S+\s*|\s+", text):
        yield chunk
        time.sleep(0.01)


# ---------------- Chat UI ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer = get_answer(st.session_state.messages[-20:])
            st.write_stream(typing_effect(answer))
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.session_state.messages.pop()  # keep history valid
