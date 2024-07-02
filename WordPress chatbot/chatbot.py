from flask import Flask, request, jsonify, render_template, redirect, url_for
import requests
from bs4 import BeautifulSoup as bs
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory




app = Flask(__name__)

scraped_text = ""
memory = ConversationBufferMemory(return_messages=True)  # Initialize ConversationBufferMemory with return_messages=True

# Functions for web scraping, text preprocessing, chunking, and embedding
def text_from_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad requests
        soup = bs(response.content, 'html.parser')

        # Removing js and css code
        for script in soup(["script", "style"]):
            script.extract()

        # Extract text content
        text = soup.get_text()

        # Cleaning the text (remove extra whitespaces, etc.)
        text_cleaning = text.strip().replace('\n', ' ')
        cleaned_text = re.sub(r"(\w)-\n(\w)", r"\1\2", text_cleaning)

        return cleaned_text
    except Exception as e:
        return f"Error: {e}"



def chunk_text(text):
    '''splitting the corpus into small chunks because LLM have limited context window. 
    Splitting text into chunks ensures each chunk fits within this window for better understanding and processing.
    Here we are using RecursiveCharacterTextSplitter from langchain
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200,
    )
    texts = text_splitter.split_text(text)
    return texts

def embedding(texts):
    '''
    Then perform the vectorization on those chunks and convert into embedding and stored in vectorstore 
    here im using huggingfaceebmbedding with the model which i used to perform retrieve data 
    and used faiss vectorstore to store the vectors 
    FAISS demonstrates exceptional proficiency in handling high-dimensional data with remarkable speed and efficiency.
    '''
    embeddings = HuggingFaceEmbeddings(model_name='HuggingFaceH4/zephyr-7b-beta')
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

# LLM and chain setup
llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token='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')
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant integrated into a WordPress website, powered by a Retrieval-Augmented Generation (RAG) model. Your task is to retrieve relevant information from a large knowledge base and use that information to generate a final answer.

When given a question, follow these steps:

1. Analyze the Question: Identify the key information needed to answer the question.
2. Retrieve Information: Retrieve relevant information from the knowledge base that could help answer the question.
3. Chain of Thought Reasoning: Walk through the reasoning process step-by-step to arrive at a final answer. This includes:
    - Stating the initial question
    - Listing the relevant information retrieved from the knowledge base
    - Explaining how the retrieved information helps answer the question
    - Connecting the dots and drawing logical conclusions
    - Stating the final answer
    

Example Chain of Thought:

Question: How do I install a plugin on my WordPress site?

Retrieved Information:
- To install a plugin, you need to access the WordPress dashboard.
- Navigate to the "Plugins" section and click "Add New".
- You can search for plugins in the search bar or upload a plugin file from your computer.
- After finding the desired plugin, click "Install Now" and then "Activate" to enable the plugin.

Chain of Thought:
The question asks how to install a plugin on a WordPress site. From the retrieved information, we learn that you need to access the WordPress dashboard and navigate to the "Plugins" section. Within this section, you can either search for plugins or upload one from your computer. Once you find the desired plugin, you should click "Install Now" and then "Activate" to enable it. By following these steps, you can successfully install and activate a plugin on your WordPress site.

Final Answer: To install a plugin on your WordPress site, go to the WordPress dashboard, navigate to "Plugins" > "Add New", search for the plugin or upload a plugin file, click "Install Now", and then "Activate" to enable the plugin.

By walking through the chain of thought, your responses will be more transparent, allowing users to understand your reasoning process. Please use this approach when answering questions.


Answer the following question based only on the provided context.

<context>
{context}
</context>
Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = None
retrieval_chain = None

@app.route("/", methods=["GET", "POST"])
def index():
    global memory, retriever, retrieval_chain
    if request.method == "POST":
        if "url" in request.form:
            url = request.form["url"]
            scraped_text = text_from_website(url)
            texts = chunk_text(scraped_text)
            vector_store = embedding(texts)
            retriever = vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            message = "URL content scraped and processed successfully!"
        elif "query" in request.form:
            query = request.form["query"]
            response = retrieval_chain.invoke({"input": query})
            answer = response["answer"]

            start_marker = "the following question based only on the provided context"
            start_index = answer.find(start_marker)

            if start_index != -1:
                generated_output = answer[start_index + len(start_marker):].strip()
                formatted_output = "\n".join(line.strip() for line in generated_output.splitlines() if line.strip())
            else:
                formatted_output = "Start marker not found in the generated text."

            memory.save_context({"input": query}, {"output": formatted_output})
            message = "Query processed successfully!"

        return redirect(url_for("index"))

    return render_template("index.html", memory=memory)

if __name__ == "__main__":
    app.run(debug=True)
