#importing pipecone modules
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Importing Langchain modules
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import Ollama
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.question_answering import load_qa_chain

# Importing other modules
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


## Reading the documents
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    doc = file_loader.load()
    return doc

# Convert the document into chunks

def doc_to_chunks(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split(docs)
    return docs

# Embedding techniques of Ollama

## Coine similarity retreive the results
def retreive_query(query, index, k=2):
    matching_results = index.similarity_search(query, k=k)
    return matching_results

## Search the answer from vector DB
def retreive_answer(query, index):
    llm = Ollama(model='llama3-chatqa')
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    doc_search = retreive_query(query, index)
    response = chain.run(input_documents=doc_search, question=query)
    return response





# Main logic beguns here

def main(doc_path, query, indexName=None, dims=None, metrics=None, access_key=None):
    global vectorstore_from_docs
    doc = read_doc(doc_path)
    # index_name = "langchain1"
    os.environ['PINECONE_API_KEY'] = access_key
    pc = Pinecone(api_key=access_key)
    index = any(index['name'] == indexName for index in pc.list_indexes().indexes)
    if index:
        embeddings = OllamaEmbeddings()
        vectorstore_from_docs = PineconeVectorStore.from_documents(
            doc,
            index_name=indexName,
            embedding=embeddings
            )
        answer = retreive_answer(query, vectorstore_from_docs)
        return answer
    else:
        pc.create_index(name=indexName,
                        dimension=4096,
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                            )
                        )

    embeddings=OllamaEmbeddings()
    vectorstore_from_docs = PineconeVectorStore.from_documents(
            doc,
            index_name=indexName,
            embedding=embeddings
        )
    answer = retreive_answer(query, vectorstore_from_docs)
    return answer



st.title("Pinecone Q&A Chatbot")
token = st.text_input("api-key", key="access_key", type="password")
directory_path = st.text_input("Directory Path", key="input")
query_input = st.text_input("query to search", key="query")
index_nm = st.text_input("index name", key="index")


submit = st.button("Submit")




if submit and directory_path and query_input and index_nm and token:
    st.write(main(directory_path, query_input, indexName=index_nm, access_key=token))
