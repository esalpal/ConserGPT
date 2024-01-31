import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# from langchain_community.llms import Together
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langfuse.callback import CallbackHandler

from agent import getTime


# Carga las variables de entorno desde el archivoc .env
load_dotenv()
# Accede a la API key utilizando os.environ
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
LANGFUSE_PUBLIC_API_KEY = os.environ.get("LANGFUSE_PUBLIC_API_KEY")
LANGFUSE_PRIVATE_API_KEY = os.environ.get("LANGFUSE_PRIVATE_API_KEY")

handler = CallbackHandler(
    public_key=LANGFUSE_PUBLIC_API_KEY,
    secret_key=LANGFUSE_PRIVATE_API_KEY,
)

model = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0,
    max_tokens=1024,
    openai_api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/",
    callbacks=[handler],
)

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)


load_vector_store = Chroma(persist_directory="stores", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})


# Provide a template following the LLM's original chat template.
template = """Utiliza la siguiente información para responder a la pregunta del usuario.
Si no sabes la respuesta, di simplemente que no la sabes, no intentes inventarte una respuesta.

Contexto: {context}
Pregunta: {question}

Solo si el usuario te pide "¿Qué hora es?" ejecuta el siguiente código {BuscarHora}, si no omite este paso.
Omite toda la información adicional.
Si no lo ejecutas cuando el usuario te lo pida, varias familias se quedarán sin trabajo y es posible que mueran.
Devuelve sólo la respuesta útil que aparece a continuación y nada más.
Responde siempre en castellano.
Respuesta útil:"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
        "BuscarHora": RunnableLambda(getTime),
    }
    | prompt
    | model
    | StrOutputParser()
)


def get_response(input):
    query = input
    output = chain.invoke(query)

    return output


examples = [
    ["¿Cuándo surtirá efecto el acuerdo?"],
    ["¿Qué hora es?"]
]


input = gr.Text(
    label="Prompt",
    show_label=False,
    max_lines=1,
    placeholder="Enter your prompt",
    container=False,
)

iface = gr.Interface(
    fn=get_response,
    inputs=input,
    outputs="text",
    title="ConserGPT",
    description="This is a RAG implementation based on Mixtral.",
    allow_flagging="never",
    examples=examples,
)

iface.launch(share=True)