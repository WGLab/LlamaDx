import os, json
import streamlit as st
from groq import Groq
from tempfile import NamedTemporaryFile
import base64
from utils import *
from LlamaDx import *
from LlamaDxRAG import *
HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 5001))
DISEASE_DOCS_DIR = './rag_diseases'
GENETICS_DOCS_DIR = './rag_genetics'
DISEASE_CHROMA = './chroma_diseases'
GENETICS_CHROMA = './chroma_genetics'
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

# initialize the RAG systems
text_rag = LlamaDxRAG(docs_dir=DISEASE_DOCS_DIR, chroma_dir = DISEASE_CHROMA, memory_bank_id='disease_id')
genetics_rag = LlamaDxRAG(docs_dir=GENETICS_DOCS_DIR, chroma_dir = GENETICS_CHROMA, memory_bank_id='genetics_id')

# initialize the chat history as streamlit session state of not present already

def clear_all():
    st.session_state.chat_history = []
    st.session_state.uploaded_file = None
    st.session_state.image_path = None


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_file" in st.session_state:
    st.session_state.uploaded_file = None
if "image_path" in st.session_state:
    st.session_state.image_path = None

# streamlit page title
st.title("🦙 LlamaDx Chatbot")
# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
# set header
st.header("Please upload an image")
# upload file
file = st.file_uploader("", type=["jpeg", "jpg", "png"], key=st.session_state["uploader_key"])

if file:
    # display image
    st.image(file, use_container_width=True)
    st.session_state.uploaded_file = file.getvalue()
    with NamedTemporaryFile(delete=False) as f:
        f.write(file.getvalue())
        image_path = f.name
# Display the uploaded image if available
# if st.session_state.uploaded_image:
#     st.image(st.session_state.uploaded_image, use_container_width=True)

# input field for user's message
user_prompt = st.chat_input("Ask Llama...")
if user_prompt:
    st.chat_message('user').markdown(user_prompt)
    ## check if this is the disease diagnosis question
    #history_chat = *st.session_state.chat_history
    if check_template(user_prompt):
        text, genetics, question = separate_texts(user_prompt)
        question = question.replace("|","")
        if file:
            llama_image = LlamaImage(image_path, llama_groq)
            image_text = llama_image.generate_information(st.session_state.chat_history)
            text += "\nAdditional information from the image: " + image_text
        # get rag for text
        nongenetic_contexts = text_rag.generate_documents(text, top_k = 10)
        text_question = "Based on the provided documents and background information, list all possible the medical disorders this patient is likely to develop. For each predicted condition, provide a thoughtful, detailed, and concise explanation (no more than 50 words)."
        text_response = llama_groq.generate_inference(nongenetic_contexts + "\nBackground Information: " + text + "\n" + text_question, st.session_state.chat_history)
        # get rag for genetic
        if "None" in genetics:
            genetics_response = "Not given"
        else:
            genetics = genetics.replace("|","") + ". These genes have been identified as pathogenic and causing for the disorders observed in this patient."
            genetics_contexts = genetics_rag.generate_documents(genetics, top_k = 5)
            genetics_questions = "Based on the provided documents and genetics information, list all the possible medical disorders this patient is likely to develop. For each predicted condition, provide a thoughtful, detailed, and concise explanation (no more than 50 words)."
            genetics_response = llama_groq.generate_inference(genetics_contexts + "\n" + genetics + "\n" + genetics_questions, st.session_state.chat_history)
        data_inputs = llama_groq.message_format(user_prompt, history_chat = None)
        combined_texts = "Background Information:\n" + text_response + "\n" + "Genetic Information:\n" + genetics_response + "\nFrom both information above, " + question
        response = llama_groq.generate_inference(combined_texts, history_chat=None)      
    else:
        if file:
            llama_image = LlamaImage(image_path, llama_groq, extract_question = user_prompt)
            data_inputs = llama_groq.message_format(user_prompt, history_chat = None)
            response = llama_image.generate_information(st.session_state.chat_history)
        else:
            data_inputs = llama_groq.message_format(user_prompt, history_chat = None)
            response = llama_groq.generate_inference(user_prompt, st.session_state.chat_history)
    if isinstance(data_inputs, dict):
        st.session_state.chat_history.append(data_inputs)
    else:
        st.session_state.chat_history.extend(data_inputs)
    st.session_state.chat_history.append({"role":"assistant", "content":response})

    # display the LLM's response
    with st.chat_message('assistant'):
        st.markdown(response)

# Clear all uploaded images and context
if st.button("Clear All", on_click=clear_all):
    pass
#st.button("Clear All", on_click=clear_all)