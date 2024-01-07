import os
import glob

import streamlit as st

# used to create the memory
from langchain.memory import ConversationBufferMemory

# used to create the retriever
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# used to create the prompt template
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder

# used to create the agent executor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor

# add a heading for your app.
st.header("Chat with LLM Agents 💬 📚")

# Initialize the memory
# This is needed for both the memory and the prompt
memory_key = "history"

if os.path.isdir("data"):
    pass
else:
    os.mkdir("data")

if 'db' not in st.session_state:
    st.session_state['db'] = None

if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None

if "memory" not in st.session_state.keys():
    st.session_state.memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)

# Initialize the chat message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the uploaded article(s)!"}
    ]

# Clear data folder
def clear_data():
    if len(os.listdir('./data')) > 0:
        data_files = glob.glob('./data/*')
        for f in data_files:
            os.remove(f)
        print('data cleared')

def save_uploadedfile(uploaded_file):
     with open(os.path.join("data",uploaded_file.name),"wb") as f:
         f.write(uploaded_file.getbuffer())
     return st.success("Saved File:{}".format(uploaded_file.name))

os.environ["OPENAI_API_KEY"] = st.secrets.openai_key

# create the document database
def load_data():
    loader = DirectoryLoader("data/")
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    # [data_index.metadata['source'] for data_index in data]
    return db

# File upload and save
uploaded_file_ls = st.file_uploader('Upload an article', accept_multiple_files=True, type=['txt','pdf','csv','docx','xlsx'])
# for uploaded_file in uploaded_file_ls:
#     save_uploadedfile(uploaded_file)
db_file_name_ls = os.listdir("data")    
uploaded_file_name_ls = [uploaded_file.name for uploaded_file in uploaded_file_ls]
to_upload_file_name_ls = list(set(uploaded_file_name_ls)-set(db_file_name_ls))
to_remove_file_name_ls = list(set(db_file_name_ls)-set(uploaded_file_name_ls))
print('db_file_name_ls:',db_file_name_ls)
print('uploaded_file_name_ls:',uploaded_file_name_ls)
print('to_upload_file_name_ls:',to_upload_file_name_ls)
print('to_remove_file_name_ls:',to_remove_file_name_ls)
for uploaded_file in uploaded_file_ls:
    save_uploadedfile(uploaded_file)
for file_name in to_remove_file_name_ls:
    os.remove(f"data/{file_name}")
# check database and uploaded file names
# check db index name and database

# if len(uploaded_file_ls) > 0:
#     print(uploaded_file_ls)
#     clear_data()
#     for uploaded_file in uploaded_file_ls:
#         save_uploadedfile(uploaded_file)
#     st.session_state['db'] = None
#     uploaded_file_ls = []

if (len(os.listdir('data'))>0) and ((len(to_upload_file_name_ls)>0)|(len(to_remove_file_name_ls)>0)):
    db = load_data()
    st.session_state['db'] = db
    # instantiate the database retriever
    retriever = db.as_retriever()
    st.session_state['retriever'] = retriever
    print(db.index)

if st.session_state['retriever'] is not None:
    tool = create_retriever_tool(
        st.session_state['retriever'],
        "search_doc",
        "Searches and returns documents.",
    )
    tools = [tool]

    # instantiate the large language model
    llm = ChatOpenAI(temperature = 0)

    # define the prompt
    system_message = SystemMessage(
            content=(
                "Do your best to answer the questions. "
                "Feel free to use any tools available to look up "
                "relevant information, only if neccessary"
            )
    )
    prompt_template = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
        )

    # instantiate agent
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

    # Prompt for user input and display message history
    if prompt := st.chat_input("Your article related question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Pass query to chat engine and display response
    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent_executor({"input": prompt})
                st.write(response["output"])
                message = {"role": "assistant", "content": response["output"]}
                st.session_state.messages.append(message) # Add response to message history
                print(st.session_state.messages)

# define the retriever tool
# @tool
# def tool(query):
#     "Searches and returns documents regarding the uploaded article(s)"
#     docs = st.session_state['retriever'].get_relevant_documents(query)
#     limit_len_docs = []
#     for doc in docs:
#         limit_len_docs.append(doc[:1000])
#     return docs

# tools = [tool]
