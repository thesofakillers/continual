import gradio as gr

from harness import model_harness

def echo(message, history):
    return message

def respond_to_user(message: str, history: list[dict[str, str]]) -> str:

    messages = history.copy()
    messages.append({"role": "user", "content": message})

    model_response = model_harness(messages)

    return model_response


demo = gr.ChatInterface(fn=respond_to_user, type="messages", examples=["hello", "hola", "merhaba"], title="Echo Bot")
demo.launch()
