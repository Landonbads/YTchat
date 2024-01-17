from youtube_transcript_api import YouTubeTranscriptApi
import chromadb
from chromadb.config import Settings
import pandas as pd
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import numpy as np
import tiktoken

class Transcript:
    chroma_client = chromadb.EphemeralClient(settings=Settings(allow_reset=True)) # shared chroma DB client

    def __init__(self,video_url,openAI_client,openAI_key):
        self.video_url = video_url
        self.api_key = openAI_key
        self.openAI_client = openAI_client
        self.text_transcript = ""
        self.chroma_collection = None

    def get_transcript(self):
        try:
            # check if URL is for a youtube short
            if self.video_url.find("shorts") != -1:
                # Extract the video ID from the URL
                video_id = self.video_url.split("shorts/")[-1]
            else:
                # Extraction for normal video URL
                video_id = self.video_url.split('watch?v=')[-1]
            # Fetch the transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            # Change transcript time to minutes and seconds
            transcript = self.convert_time(transcript)
            # Separate the transcript into bigger chunks
            transcript = self.join_data(transcript)
            for text in transcript:
                self.text_transcript += f"[{text['start']}]: {text['text']}\n"
            return self.text_transcript
        except Exception as e:
            print('Error:', e)
        
    # method to separate/join data so that a bigger chunk of data is retrieved from DB and sent as context to LLM
    def join_data(self,transcript,max_duration=30):
        joined_data = []
        current_text = ""
        current_start = 0
        current_duration = 0

        for seg in transcript:
            # check if adding new duration makes duration over 30 seconds
            if current_duration + seg['duration'] <= max_duration:
                # append new text to current text
                if current_text:
                    current_text += " " + seg['text'] 
                else:
                    current_text += seg['text']

                current_duration += seg['duration']
                # if it's the first append in the sequence add the start time
                if current_duration == seg['duration']:
                    current_start = seg['start']
            else:
                joined_data.append({'text': current_text, 'start': current_start, 'duration': current_duration})
                current_text = seg['text']
                current_start = seg['start']
                current_duration = seg['duration']

            # append end segment(s) if not empty
        if current_text:
            joined_data.append({'text': current_text, 'start': current_start, 'duration': current_duration})

        return joined_data
    
    # function to convert seconds to HH:MM:SS format
    def convert_time(self,transcript):
        for seg in transcript:
            total_seconds = seg['start']
            hours, remainder = divmod(int(total_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            seg['start'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return transcript
    
    # OpenAI embedding batch requests require input to be < 8k tokens. 
    # this allows querying video transcripts with > 8k tokens of text
    def split_batches(self, text_transcript):
        max_tokens = 8000
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text_transcript)
        # Split tokens into chunks of max_tokens
        token_blocks = [tokens[i: i+max_tokens] for i in range(0,len(tokens), max_tokens)]
        text_blocks = [encoding.decode(block) for block in token_blocks]
        return text_blocks
    
    def embed_transcript(self,text_transcript):
        self.chroma_client.reset()
        embedding_model = "text-embedding-ada-002" 
        # dictionary containing text and embedded vector for each piece of text
        df = {'text': [], 'text_vector': []}
        batches = []
        # split text into blocks containing < 8k tokens
        text_blocks = self.split_batches(text_transcript)
        # breakup each block of text into smaller snippets
        for text in text_blocks:
            df['text'] += [line for line in text.rstrip("\n").split("\n")]
            batches.append([line for line in text.rstrip("\n").split("\n")])
        # batch embedding request to openAI to put into vector DB
        for batch in batches:
            embeddings = self.openAI_client.embeddings.create(
                input = batch, model=embedding_model)
            for embed in embeddings.data:
                df['text_vector'].append(embed.embedding)

        # create the to be used embedding function that will be used to query the chroma DB
        embedding_function = OpenAIEmbeddingFunction(api_key=self.api_key, model_name=embedding_model)
        # create the chroma collection that will store the embedded transcript
        transcript_collection = self.chroma_client.create_collection(
            name='transcript_content',
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"} # using cosine similarity
        )
        # convert dictionary to dataframe
        df = pd.DataFrame(data=df)
        df['id'] = np.arange(df.shape[0]) # add id's col, 0 through n for df with n rows
        # set id to be a string
        df['id'] = df['id'].apply(str)
        # add the lists to the collection
        transcript_collection.add(
            ids=df.id.to_list(),
            embeddings=df.text_vector.tolist()
        )
        # convert dictionary to dataframe
        df = pd.DataFrame(data=df)
        self.chroma_collection = transcript_collection
        # return dataframe with ids, text, and vectors
        return df
    
    # function to query the embedded database
    def query_collection(self, query, max_results, dataframe):
        results = self.chroma_collection.query(query_texts=query, n_results=max_results, include=['distances']) 
        df = pd.DataFrame({
                    'collection_id':results['ids'][0], 
                    'score':results['distances'][0],
                    'content': dataframe[dataframe.id.isin(results['ids'][0])]['text'],
                    })
        
        # return the 3 matches with the highest similarity score
        print(df)
        return df.tail(15)
        