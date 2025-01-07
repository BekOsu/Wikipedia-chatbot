import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# OpenAI API key and model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTORSTORE_PATH = "./wiki_embeddings"


def suggest_questions_from_vectorstore(query="Topics to explore", k=10):
    """
    Suggests unique topics based on the vectorstore.
    Args:
        query (str): The query to search for in the vectorstore.
        k (int): Number of results to retrieve.
    Returns:
        list: A list of unique topics from the vectorstore.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=k)
    topics = list({doc.page_content.strip() for doc in results})  # Use a set to ensure uniqueness
    return topics


def generate_questions_from_topics(topics, max_templates=2):
    """
    Generates unique questions based on topics using predefined templates.
    Args:
        topics (list): List of topics.
        max_templates (int): Maximum number of templates to apply per topic.
    Returns:
        list: A list of unique generated questions.
    """
    question_templates = [
        "What is {}?",
        "Can you explain {}?",
        "What are the main points about {}?",
        "Why is {} important?",
    ]

    questions = []
    for topic in topics:
        for template in question_templates[:max_templates]:
            questions.append(template.format(topic))

    return list(set(questions))  # Remove duplicates from the questions


def save_questions_to_file(questions, file_path="generated_questions.txt"):
    """
    Saves the generated questions to a text file.
    Args:
        questions (list): List of questions.
        file_path (str): File path to save the questions.
    """
    with open(file_path, "w") as file:
        for question in questions:
            file.write(question + "\n")
    print(f"Questions saved to {file_path}")


def main():
    """
    Main function to generate and display questions.
    """
    # Allow the user to input a query
    query = input("Enter a query to explore topics (default: 'Key topics in the articles'): ").strip()
    if not query:
        query = "Key topics in the articles"

    # Suggest topics
    topics = suggest_questions_from_vectorstore(query=query)
    if not topics:
        print("No topics found for the query.")
        return

    print("\nSuggested Topics:")
    for topic in topics:
        print(f"- {topic}")

    # Generate questions
    questions = generate_questions_from_topics(topics)
    print("\nGenerated Questions:")
    for question in questions:
        print(f"- {question}")

    # Save questions to a file
    save_questions_to_file(questions)


if __name__ == "__main__":
    main()
