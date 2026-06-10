import os
import gradio as gr
from tool_manager import handle_tool_calls
from tools import tools
from system_prompt import system_prompt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

groq_api_key = os.getenv('GROQ_API_KEY')

groq = OpenAI( 
    base_url="https://api.groq.com/openai/v1", 
    api_key=groq_api_key
)

MODEL = 'llama-3.3-70b-versatile'

# Removed 'self' parameter
def chat(message, history):
    # Clean the history to remove Gradio-specific metadata that Groq rejects
    clean_history = [
        {"role": msg["role"], "content": msg["content"]} 
        for msg in history
    ]
    
    # Build the messages payload using the clean history
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
            
    return response.choices[0].message.content

gr.ChatInterface(chat).launch()