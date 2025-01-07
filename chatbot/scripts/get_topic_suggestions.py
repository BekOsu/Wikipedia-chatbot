import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# OpenAI API key and model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VECTORSTORE_PATH = "./wiki_embeddings"


def clean_topic(topic):
    """
    Cleans and filters a single topic string.
    Args:
        topic (str): The topic string to clean.
    Returns:
        str or None: Cleaned topic string or None if the topic is invalid.
    """
    topic = topic.strip().replace("\n", " ").replace("\r", "")  # Remove newlines and strip whitespace
    if len(topic.split()) <= 2:  # Skip topics with very few words
        return None
    if any(keyword in topic.lower() for keyword in ["pages", "books", "sources", "other websites"]):  # Skip irrelevant topics
        return None
    return topic


def suggest_questions_from_vectorstore(query="Topics to explore", k=10):
    """
    Suggests unique topics based on the vectorstore.
    Args:
        query (str): The query to search for in the vectorstore.
        k (int): Number of results to retrieve.
    Returns:
        tuple: A tuple containing raw topics and cleaned topics.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=k)

    # Raw topics
    raw_topics = [doc.page_content.strip() for doc in results]

    # Clean and filter topics
    cleaned_topics = list({clean_topic(topic) for topic in raw_topics})
    cleaned_topics = [topic for topic in cleaned_topics if topic]  # Remove None values

    # Keep raw topics that failed cleaning but are still useful
    missing_cleaned_topics = [topic for topic in raw_topics if topic not in cleaned_topics]

    return raw_topics, cleaned_topics, missing_cleaned_topics


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


def main():
    """
    Main function to generate and display questions.
    """
    query = input("Enter a query to explore topics (default: 'Key topics in the articles'): ").strip()
    if not query:
        query = "Key topics in the articles"

    # Suggest topics
    raw_topics, cleaned_topics, missing_cleaned_topics = suggest_questions_from_vectorstore(query=query)

    print("\nRaw Topics (Before Cleaning):")
    for topic in raw_topics:
        print(f"- {topic}")

    print("\nCleaned Topics (After Cleaning):")
    for topic in cleaned_topics:
        print(f"- {topic}")

    print("\nTopics Skipped or Missing in Cleaned Topics:")
    for topic in missing_cleaned_topics:
        print(f"- {topic}")

    if not cleaned_topics:
        print("No topics found for the query.")
        return

    # Generate questions
    questions = generate_questions_from_topics(cleaned_topics)
    print("\nGenerated Questions:")
    for question in questions:
        print(f"- {question}")

    # Save questions to a file
    save_questions_to_file(questions)


def save_questions_to_file(questions, filename="generated_questions.txt"):
    """
    Saves questions to a text file.
    """
    with open(filename, "w") as f:
        for question in questions:
            f.write(f"{question}\n")
    print(f"Questions saved to {filename}")


if __name__ == "__main__":
    main()
