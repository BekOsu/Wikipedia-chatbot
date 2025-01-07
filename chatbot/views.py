import os
from django.shortcuts import render
from django.http import JsonResponse
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# OpenAI API key and model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_API_MODEL", "gpt-4")
VECTORSTORE_PATH = "./wiki_embeddings"


# Global retriever and QA chain
def load_retriever():
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vectorstore.as_retriever()
    except Exception as e:
        raise RuntimeError(f"Error loading retriever: {e}")


retriever = load_retriever()
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# Chat view
def chat(request):
    if request.method == "POST":
        question = request.POST.get("question", "").strip()
        if not question:
            return JsonResponse({"error": "Please provide a valid question."}, status=400)

        try:
            result = qa_chain.invoke({"question": question})
            answer = result.get("answer", "No answer found.")
            return JsonResponse({"answer": answer})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return render(request, "chatbot/chat.html")
