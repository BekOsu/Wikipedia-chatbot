import os
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from .scripts.chat_with_wikipedia import get_qa_chain

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


@csrf_exempt
def chat_view(request):
    if request.method == "GET":
        # Render a simple HTML page with the chatbot interface
        from django.shortcuts import render
        return render(request, "chat_with_bot.html")  # Ensure this template exists

    if request.method == "POST":
        # Handle the chatbot question
        question = request.POST.get("question")
        if not question:
            return JsonResponse({"error": "No question provided."}, status=400)

        result = qa_chain.invoke({"question": question})
        answer = result.get("answer", "Sorry, I couldn't find an answer.")
        return JsonResponse({"answer": answer})

    return JsonResponse({"error": "Invalid request method."}, status=405)

