
from flask import Flask, render_template, request
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import params
from openai import OpenAI as pipgpt
import warnings
from pprint import pprint

app = Flask(__name__)
client = pipgpt(api_key=params.openai_api_key)
# Initialize MongoDB python client
client1 = MongoClient(params.mongodb_conn_string)
collection = client1[params.db_name][params.collection_name]

# Initialize vector store
vectorStore = MongoDBAtlasVectorSearch(
    collection, OpenAIEmbeddings(openai_api_key=params.openai_api_key), index_name=params.index_name
)

# Initialize OpenAI
llm = OpenAI(openai_api_key=params.openai_api_key, temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

# Initialize compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorStore.as_retriever()
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query_text = request.form['query']
    print("\nYour question:")
    print("-------------")
    print(query_text)

    # Perform vector store search
    print("---------------")
    docs = vectorStore.max_marginal_relevance_search(query_text, K=1)

    # Perform contextual compression
    print("\nAI Response:")
    print("-----------")
    compressed_docs = compression_retriever.get_relevant_documents(query_text)
    pprint(compressed_docs)
    complete_prompt="""you are an assistant for semiconductor responses.
     Based on the provided documents and your own knowledge generate concise responses for input questions
      QUESTION:  """ + query_text + "\n"
    for i in range(len(compressed_docs)):
        complete_prompt+="Document " + str(i+1) + ": " + compressed_docs[i].page_content + "\n"

    messages = [{"role": "user", "content": complete_prompt}]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature= 0.3,
        max_tokens= 500
    
    )
    print(completion)
    return render_template('result.html', query=query_text, response=completion.choices[0].message.content)

if __name__ == '__main__':
    app.run(debug=True)
