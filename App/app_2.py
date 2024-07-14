import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import set_global_service_context, ServiceContext
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.readers.file import PyMuPDFReader
from pathlib import Path

model_name = '../RAG_finetune/LLama2-7b-OS'
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

@st.cache_resource
def get_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, quantization_config=bnb_config)
    return tokenizer, model

tokenizer, model = get_tokenizer_model()

system_prompt = """[INST] <>
You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.

Please do not use <ANSWER> tag. 
Your goal is to provide answers relating to the Operating systems.<>
"""

query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=1024,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    model=model,
    tokenizer=tokenizer
)

embeddings = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="../Embeddings/bge-large-matryoshka/")
)

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
set_global_service_context(service_context)

storage_context = StorageContext.from_defaults(persist_dir="../Data/Index")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)

st.title('🦙 Llama Chat')
prompt = st.text_input('Input your prompt here')

if prompt:
    placeholder = st.empty()
    full_response = ""
    
    response = query_engine.query(prompt)
    response_text = ""
    for text in response.response_gen:
        if "<ANSWER>" in text:
            break
        full_response += text
        response_text += text
        placeholder.markdown(full_response)

    with st.expander('Response Object'):
        st.write(response)

    with st.expander('Source Text'):
        st.write(response.get_formatted_sources())
