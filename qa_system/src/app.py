import os
import io
import yaml
import functools
from fastapi.responses import Response
from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002',
                              deployment='text-embedding-ada-002',
                              chunk_size=1)

db = ElasticVectorSearch(
    elasticsearch_url="http://vectordatabase:9200",
    index_name="elastic-index",
    embedding=embeddings,
)

llm = OpenAI(engine='text-davinci-003', temperature=0)

prompt_template = """Você auxilia os cliente da empresa Dr. Consulta com dúvidas sobre a empresa e seus processos. Seja breve em suas respostas. Responda APENAS com os fatos listados na lista de fontes abaixo. Se não houver informações suficientes abaixo, diga que não sabe. Não gere respostas que não usem as fontes abaixo. Se fazer uma pergunta esclarecedora ao usuário ajudar, faça a pergunta.

{context}

Pergunta: {question}
Resposta:"""

prompt_template = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": prompt_template}

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(), 
    chain_type_kwargs=chain_type_kwargs
)

app = FastAPI()

# additional yaml version of openapi.json
@app.get('/openapi.yaml', include_in_schema=False)
@functools.lru_cache()
def read_openapi_yaml() -> Response:
    openapi_json= app.openapi()
    yaml_s = io.StringIO()
    yaml.dump(openapi_json, yaml_s)
    return Response(yaml_s.getvalue(), media_type='text/yaml')


@app.get("/")
def index():
    return {
        "message": "Make a post request to /ask to ask questions about Meditations by Marcus Aurelius"
    }


@app.post("/ask")
def ask(query: str):
    """Gets a user query as input and use it to answer a question usar a knowledge base as reference"""
    response = qa.run(query)
    return {
        "response": response,
    }
