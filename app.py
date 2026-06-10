import os
import gradio as gr
from src.tool_manager import handle_tool_calls
from src.tools import tools
from src.system_prompt import system_prompt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

groq_api_key = os.getenv('GROQ_API_KEY')

groq = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

MODEL = 'llama-3.3-70b-versatile'

def chat(message, history):
    clean_history = []
    for msg in history:
        if isinstance(msg, dict):
            clean_history.append({"role": msg["role"], "content": msg["content"]})
        else:
            user_msg, bot_msg = msg
            clean_history.append({"role": "user", "content": user_msg})
            if bot_msg:
                clean_history.append({"role": "assistant", "content": bot_msg})

    messages = [{"role": "system", "content": system_prompt}] + clean_history + [{"role": "user", "content": message}]

    done = False
    while not done:
        response = groq.chat.completions.create(model=MODEL, messages=messages, tools=tools)

        if response.choices[0].finish_reason == "tool_calls":
            msg_obj = response.choices[0].message
            tool_calls = msg_obj.tool_calls
            results = handle_tool_calls(tool_calls)
            messages.append(msg_obj)
            messages.extend(results)
        else:
            done = True

    # Append to history and return both outputs
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response.choices[0].message.content})
    return "", history


css = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

body, .gradio-container {
    background-color: #0a0a0f !important;
    font-family: 'Inter', sans-serif !important;
}

#header {
    text-align: center;
    padding: 2rem 0 1rem;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 1rem;
}

#header h1 {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 400 !important;
    color: #00ffe7 !important;
    text-shadow: 0 0 20px #00ffe7, 0 0 40px #00ffe799;
    letter-spacing: 0.08em;
    margin: 0 !important;
}

#header p {
    color: #7c6f9f !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em;
    margin-top: 0.4rem !important;
    text-transform: uppercase;
}

#chatbox {
    background: #0d0d17 !important;
    border: 1px solid #1e1e3a !important;
    border-radius: 8px !important;
}

#chatbox .message.user {
    background: #1a0a2e !important;
    border: 1px solid #6e2fff44 !important;
    color: #e2d9f3 !important;
    border-radius: 8px 8px 2px 8px !important;
}

#chatbox .message.bot {
    background: #071a1a !important;
    border: 1px solid #00ffe733 !important;
    color: #c8faf5 !important;
    border-radius: 8px 8px 8px 2px !important;
}

#msg-input textarea {
    background: #0d0d17 !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 6px !important;
    color: #e2d9f3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s;
}

#msg-input textarea:focus {
    border-color: #00ffe7 !important;
    box-shadow: 0 0 0 2px #00ffe722 !important;
    outline: none !important;
}

#msg-input textarea::placeholder {
    color: #4a4a6a !important;
}

#send-btn {
    background: linear-gradient(135deg, #00ffe7, #6e2fff) !important;
    border: none !important;
    border-radius: 6px !important;
    color: #0a0a0f !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em;
    transition: opacity 0.2s, transform 0.1s !important;
}

#send-btn:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

#clear-btn {
    background: transparent !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 6px !important;
    color: #7c6f9f !important;
    font-size: 0.8rem !important;
    transition: border-color 0.2s, color 0.2s !important;
}

#clear-btn:hover {
    border-color: #ff4f8b !important;
    color: #ff4f8b !important;
}

#chatbox ::-webkit-scrollbar { width: 4px; }
#chatbox ::-webkit-scrollbar-track { background: transparent; }
#chatbox ::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 4px; }
"""

with gr.Blocks() as demo:

    with gr.Column(elem_id="header"):
        gr.Markdown("# PROFESSIONALLY YOU")
        gr.Markdown("career advisor · resume strategist · interview coach")

    chatbot = gr.Chatbot(
        height=480,
        elem_id="chatbox",
        show_label=False,
    )

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Ask me anything...",
            scale=8,
            show_label=False,
            container=False,
            elem_id="msg-input",
        )
        submit_btn = gr.Button("Send →", scale=1, elem_id="send-btn")

    clear_btn = gr.Button("Clear conversation", elem_id="clear-btn")

    msg_input.submit(fn=chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
    submit_btn.click(fn=chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
    clear_btn.click(fn=lambda: ("", []), outputs=[msg_input, chatbot])

if __name__ == "__main__":
    demo.launch(css=css)