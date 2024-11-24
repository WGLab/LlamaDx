import base64
from LlamaDxRAG import *
import os, json
import streamlit as st
from groq import Groq
from utils import *
from LlamaDx import *
HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 1003))
class LlamaImage():
    def __init__(self, image_url, model, extract_question = "Please extract any medical phenotypes (signs and symptoms) from this image. Be concise, do not repeat, and do not generate random details."):
        self.image_url = image_url
        self.extract_question = extract_question
        self.model = model
    def encode_image(self):
        with open(self.image_url, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    def message_format(self):
        self.messages =   {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.extract_question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":  f"data:image/jpg;base64,{self.encode_image()}"
                        }
                    }
                ]
            }
        return self.messages
    def generate_information(self, history_chat):
        self.messages = self.message_format()
        return self.model.generate_inference(self.messages, history_chat, data_type = 'image')

class LlamaGroq():
    def __init__(self, client, model = 'llama-3.2-11b-vision-preview'):
        self.client = client
        self.model = model
    def message_format(self, user_prompt, history_chat = None):
        messages = [{
                'role': "system",
                'content': (
                    # "You are a medical assistant. You should answer questions reliably "
                    #"and don't make up the answer. If you don't know the answers, simply say 'I don't know.'"
                    "You are a medical assistant! Answer questions accurately and responsibly. Provide reliable, evidence-based information and avoid making up details."
                )
            }]

        # Add chat history if provided
        if history_chat:
            if isinstance(history_chat, dict):
                messages = [messages, history_chat]
            else:
                messages = history_chat + messages

        # Append the user's message
        messages.append({
            "role": "user",
            "content": user_prompt
        })

        return messages
    def generate_inference(self, user_prompt, history_chat, data_type = 'text'):
        if data_type == 'text':
            messages = self.message_format(user_prompt, history_chat)
        else:
            #messages = history_chat + [user_prompt]
            messages = [user_prompt]
        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature=0.2,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return response.choices[0].message.content


class LlamaStack():
    def __init__(self):
        self.LlamaDxRAG = LlamaDxRAG(host = HOST, port = PORT)

    def generate_information(self, query, num_texts):
        return self.LlamaDxRAG.generate_documents(query, num_texts=5)

def test():
    # streamlit page configuration
    st.set_page_config(
        page_title="Llama 3.2 chat",
        page_icon = "🦙",
        layout = 'centered'
    )

    working_dir = os.path.dirname(os.path.abspath(__file__))
    config_data = json.load(open(f"{working_dir}/config.json"))

    GROQ_API_KEY = config_data['GROQ_API_KEY']
    os.environ['GROQ_API_KEY'] = GROQ_API_KEY

    client = Groq()
    llama_groq = LlamaGroq(client)