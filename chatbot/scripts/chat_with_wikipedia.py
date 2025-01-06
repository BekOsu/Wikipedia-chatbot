import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.prompt import Prompt

load_dotenv()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_API_MODEL", "gpt-4")

VECTORSTORE_PATH = "./wiki_embeddings"


def load_retriever():
    """Load the FAISS retriever from the stored vector database."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vectorstore.as_retriever()
    except FileNotFoundError:
        print(f"Vector database not found at {VECTORSTORE_PATH}. Run the embedding ingestion step first.")
        exit(1)


def get_qa_chain():
    """Initialize a ConversationalRetrievalChain for Q&A."""
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=OPENAI_API_KEY)

    retriever = load_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


if __name__ == "__main__":
    # Command-line loop for chatting with Wikipedia data
    console = Console()
    console.print("[bold]Chat with Wikipedia!")
    console.print("[bold red]-------------------")
    qa_chain = get_qa_chain()

    while True:
        default_question = "Ask Wikipedia: what is August?"
        question = Prompt.ask("Your Question", default=default_question)
        result = qa_chain({"question": question})
        console.print(f"[green]Answer: [/green]{result['answer']}")
        console.print("[bold red]-------------------")
