# Import streamlit for app dev
import streamlit as st

# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer,BitsAndBytesConfig
# Import torch for datatype attributes 
import torch
# Import the prompt wrapper...but for llama index
from llama_index.core.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
from llama_index.llms.huggingface import HuggingFaceLLM
# Bring in embeddings wrapper
from llama_index.embeddings.langchain import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Bring in stuff to change service context
from llama_index.core import set_global_service_context
from llama_index.core import ServiceContext
# Import deps to load documents 
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.readers.file import PyMuPDFReader
from pathlib import Path
import textwrap

model_name = 'meta-llama/Llama-2-7b-chat-hf'
auth_tok = ''
# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
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
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, 
    rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True,token=auth_tok) 

    return tokenizer, model
tokenizer, model = get_tokenizer_model()

# Create a system prompt 
system_prompt = """[INST] <>
You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.

Your goal is to provide answers relating to the Purchase policy.<>
"""
# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Create a HF LLM using the llama index wrapper 
llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=1024,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

# Create and dl embeddings instance  
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
)

# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
# And set the service context
set_global_service_context(service_context)

# Create PDF Loader
loader = PyMuPDFReader()
# Load documents 
documents = loader.load(file_path=Path('../Data/Os_books/UHPP.pdf'), metadata=True)

# Create an index - we'll be able to query this in a sec
index = VectorStoreIndex.from_documents(documents)

# # rebuild storage context
# storage_context = StorageContext.from_defaults(persist_dir="../Data/Index")

# # load index
# index = load_index_from_storage(storage_context)

# Setup index query engine using LLM 
query_engine = index.as_query_engine(streaming=True,similarity_top_k=3)

# Create centered main title 
st.title('🦙 Llama Chat')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
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


