# RAG (Retrieval Augmented Generation) System for Document Understanding

## Overview

This repository implements a RAG (Retrieval Augmented Generation) system designed to enhance document understanding through advanced retrieval and generation techniques. It enables users to retrieve relevant information from documents and generate informative summaries or responses, leveraging state-of-the-art AI models for improved comprehension and analysis.

## Features

- **Upload PDF File:**
  Users can upload PDF documents to the system for analysis and processing.

- **Chatbot Interface:**
  Integration of a chatbot interface that interacts with users to query documents and provide summaries or responses using the RAG system.

## Installation

To run the program, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo

2.  **Build the Docker Image:**
    Build the Docker image using the provided Dockerfile and tag it with a version number (v1 in this example):
    ```bash
    docker build -t <tag-name>:v1 .

3.  **Run the Docker Container:**
    Start a Docker container with the built image. Be sure to pass your OpenAI API key using the -e environment variable option:
    ```bash
    docker run -d -e OPENAI_API_KEY="<openai api key>" -e <tag-name>:v1

4.  **Verify Installation:**
    Confirm that the Docker container is running correctly and accessible as expected.

5.  **Prerequisites**

    Docker installed on your machine.
    An OpenAI API key obtained from OpenAI.

**Notes**
    Replace <tag-name> with your desired Docker image tag name.
    Ensure your OpenAI API key is securely stored and passed to the Docker container as shown above.
