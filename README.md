# Charles' Conversational Career Agent

An AI-driven career advisor, resume strategist, and interview coach built with Python, Gradio, and Groq. Personalized to analyze my engineering and professional background, this agent utilizes a recursive tool-calling loop with the `llama-3.3-70b-versatile` model to analyze professional documents, track custom metrics, and handle specialized career management workflows.

## 🔗 Live Demo
👉 **[Launch Live Demo App](https://huggingface.co/spaces/CharlyiE/conversational-career-agent)**

---

## 🛠️ Tech Stack & Architecture

* **Inference Engine:** Groq API (`llama-3.3-70b-versatile`) via the OpenAI Python Client.
* **User Interface:** Gradio 6.0 (Custom layout using programmatic blocks).
* **Package Management:** `uv` (Fast Python package installer and resolver).
* **Document Parsing:** `pypdf` for extracting text data from PDF resumes and CVs.
* **Environment Management:** `python-dotenv` for local credential security.

### Project Structure
```text
conversational-career-agent/
├── app.py                 # Core application entry point and Gradio UI
├── .env                   # Local environment credentials (Secret)
├── .gitignore             # Git exclusion rules (me/, .env, etc.)
├── pyproject.toml         # Project metadata and dependencies managed by uv
└── src/
    ├── tool_manager.py    # Recursive agent tool execution handler
    ├── tools.py           # Specialized tool definitions (metrics logging, ingestion)
    ├── system_prompt.py   # Core personality and behavior instructions
    ├── file_manager.py    # Local PDF parsing and document processing
    └── notify.py          # Notification integration module