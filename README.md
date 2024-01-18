# YouTube Transcript Summarizer and Query Tool

## Overview

This script uses OpenAI's API and ChromaDB to provide a way to interact with YouTube videos. It enables users to obtain concise summaries of YouTube videos or ask specific questions about the video content. The core functionality revolves around extracting video transcripts, embedding them for efficient querying, and leveraging openAI's API for generating summaries and answering questions.

## Features

- Transcript Extraction: Retrieves transcripts of YouTube videos, including standard videos and YouTube Shorts using regex.
- Transcript Processing: Converts transcript timestamps to a readable HH:MM:SS format and joins data into manageable chunks.
- AI-Powered Summarization: Utilizes OpenAI's API to generate accurate and concise summaries of the entire video transcript.
- Contextual Querying: Implements Retrieval-Augmented Generation (RAG) using ChromaDB to find the most relevant transcript snippets in response to user queries.
- Gradio App Interface: Offers an easy-to-use Gradio interface, making the tool accessible on Hugging Face Spaces.

## To use this tool with your own OpenAI API key, follow this:

**Clone the Repository**

- git clone [repository URL]

**Install Dependencies**

- cd [repository directory]
- pip install -r requirements.txt

**Configure API Key:**

- Replace the placeholder for the OpenAI API key in the code with your own key.  
  **Before:**
  `openAI_key = os.getenv('OPENAI_KEY')`  
  **After:**
  `openAI_key = 'your_openai_api_key_here'`
  
Important: Ensure that your OpenAI API key is kept secure and not exposed publicly.

**Run app.py**  

**You can test the app with whatever API credits I have left [here](https://huggingface.co/spaces/landonbd/YTchat).**
