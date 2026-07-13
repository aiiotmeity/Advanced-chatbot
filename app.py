import os
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory as ChatMessageHistory

# =====================================================================
# STEP 1: DEFINE DATA & INGEST LOCALLY (FREE EMBEDDINGS)
# =====================================================================
knowledge_base_text = """
Project Genesis is our company's secret internal AI initiative.
The budget for Project Genesis is $5,000,000 for the fiscal year 2026.
The lead researcher running the project is Dr. Sarah Jenkins. 
The strict deadline for the prototype delivery is October 14, 2026.
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_text(knowledge_base_text)

# This downloads a tiny, free math model to your computer to handle the data
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_database = Chroma.from_texts(chunks, embeddings_model)
retriever_tool = vector_database.as_retriever(search_kwargs={"k": 2})

# =====================================================================
# STEP 2: CREATE THE SYSTEM PROMPT TEMPLATE
# =====================================================================
system_instruction_string = """You are an advanced company assistant. 
Use the following pieces of retrieved context to answer the user's question. 

Retrieved Context:
{context}

Answer the user's question accurately based on the context above and your conversation history."""

prompt_blueprint = ChatPromptTemplate.from_messages([
    ("system", system_instruction_string),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# =====================================================================
# STEP 3: ASSEMBLE THE LOCAL OLLAMA BRAIN
# =====================================================================
# This targets the Llama3 model running locally on your computer machine
llm_brain = ChatOllama(model="llama3", temperature=0.3)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def execution_chain(inputs):
    retrieved_docs = retriever_tool.invoke(inputs["input"])
    formatted_context = format_docs(retrieved_docs)
    
    return prompt_blueprint.format_prompt(
        context=formatted_context,
        chat_history=inputs.get("chat_history", []),
        input=inputs["input"]
    ).to_messages()

memory_storage = ChatMessageHistory()

final_bot_chain = RunnableWithMessageHistory(
    execution_chain | llm_brain,
    lambda session_id: memory_storage,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# =====================================================================
# STEP 4: INTERACTIVE USER TERMINAL LOOP
# =====================================================================
print("\n🤖 Local Free Chatbot Loaded! Ask about 'Project Genesis' (Type 'exit' to quit)\n")

while True:
    user_message = input("You: ")
    if user_message.lower() == 'exit':
        print("Goodbye!")
        break
        
    bot_reply = final_bot_chain.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": "local_learning_session"}}
    )
    print(f"Bot: {bot_reply.content}\n")