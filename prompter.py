import os
import json
from openai import OpenAI

class Chatter:
    def __init__(self, model_name = 'deepseek-r1:8b'):
        self.model_name_all = ['deepseek-r1:8b']

        assert model_name in self.model_name_all
        self.model_name = model_name

        self.client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

    def chat_no_rag(self, user_input, messages = None):
        if messages is None:
            messages = []
        messages.append({"role": "user", "content": user_input})

        r = self.client.chat.completions.create(model=self.model_name, messages=messages)
        reply = r.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        return reply, messages
