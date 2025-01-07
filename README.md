# Chat Backend for Wikipedia Articles in Django

This project combines Django, Wikipedia data, and modern NLP tools to create a powerful chat backend capable of retrieving meaningful responses. By leveraging Wikipedia content, embeddings, and vector databases, we aim to build a scalable and cost-efficient system.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Setup Instructions](#setup-instructions)
5. [Architecture Overview](#architecture-overview)
6. [Implementation Details](#implementation-details)
7. [How to Use](#how-to-use)
8. [Future Enhancements](#future-enhancements)

---

## Introduction

Content-aware chatbots are increasingly common on the web. What used to feel like magic can now be implemented with some know-how and a little code. This tutorial walks you through setting up a Django backend for a chat application powered by Wikipedia content to help start your chatbot journey.

This project demonstrates how to:
- Efficiently process and ingest Wikipedia articles.
- Use OpenAI’s embedding models to represent text for similarity searches.
- Employ FAISS, a vector database, for fast content retrieval based on user queries.

---

## Features

- **Efficient Wikipedia Article Processing**: Extract and optimize relevant content from Wikipedia articles.
- **Advanced NLP Integration**: Use OpenAI embedding models for similarity-based searches.
- **Fast Content Retrieval**: Leverage FAISS for quick and scalable search results.
- **Cost Efficiency**: Optimize operational costs by truncating excess information and reducing unnecessary data processing.
- **Scalability**: Ensure the backend can handle increasing loads with minimal adjustments.

---

## Prerequisites

- Python 3.8+
- Django 4.x
- OpenAI API access
- FAISS library
- SQLite or any other supported database for Django

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/BekOsu/Wikipedia-chatbot.git
cd chat-backend-wikipedia
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Add your OpenAI API key to the `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
```

### 5. Setup the Database
```bash
python manage.py migrate
```

### 6. Run the Development Server
```bash
python manage.py runserver
```

---

## Architecture Overview

1. **Data Ingestion**:
    - Fetch Wikipedia articles.
    - Optimize articles by truncating unnecessary details.
    - Save processed data in the database.

2. **Embeddings**:
    - Generate embeddings for article text using OpenAI models.
    - Store embeddings in a FAISS vector database for efficient searches.

3. **Query Processing**:
    - Accept user queries via API endpoints.
    - Perform similarity searches using FAISS.
    - Return the most relevant Wikipedia articles as responses.

---

## Implementation Details

### Wikipedia Data Processing
- Use `wikipedia-api` or similar libraries to fetch and parse articles.
- Preprocess content by removing headers, footers, and irrelevant sections.
- Save cleaned data to the database.

### Text Embedding
- Utilize OpenAI’s embedding models to convert text into vector representations.
- Optimize embeddings for similarity search tasks.

### Vector Search
- Implement FAISS to store and retrieve embeddings efficiently.
- Fine-tune search parameters for accuracy and speed.

### Django Backend
- Create API endpoints for:
  - Uploading Wikipedia data.
  - Processing user queries.
- Ensure the backend is modular and adheres to best practices.

---

## How to Use

1. **Ingest Wikipedia Articles**:
    - Use the provided script to fetch and preprocess articles.
    ```bash
    python manage.py ingest_wikipedia --topics "Artificial Intelligence, Machine Learning"
    ```

2. **Query the Chatbot**:
    - Send a POST request to the `/api/query/` endpoint with your query.
    Example:
    ```json
    {
        "query": "What is artificial intelligence?"
    }
    ```

3. **Receive Responses**:
    - The API will return the most relevant articles based on the query.

---

## Future Enhancements

- Add multi-language support for Wikipedia articles.
- Integrate caching for frequently accessed queries.
- Explore advanced embedding models for improved accuracy.
- Implement user authentication and session management.

---

With this project, you’ll not only create a functional chat backend but also gain insights into building scalable and intelligent systems. Happy coding!

