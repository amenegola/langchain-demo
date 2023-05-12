from langchain.document_loaders import BSHTMLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch
from langchain.document_loaders import PyPDFDirectoryLoader

def main():    
    loader = PyPDFDirectoryLoader("src/data/")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
    )

    documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002',
                              deployment='text-embedding-ada-002',
                              openai_api_base='',
                              openai_api_type='azure',
                              openai_api_key='',
                              chunk_size=1)
    
    db = ElasticVectorSearch.from_documents(
        documents,
        embeddings,
        elasticsearch_url="http://localhost:9200",
        index_name="elastic-index",
    )
    print(db.client.info())


if __name__ == "__main__":
    main()