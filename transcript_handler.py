from youtube_transcript_api import YouTubeTranscriptApi
import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import numpy as np

class Transcript:

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
    

    def embed_transcript(self,text_transcript):
        embedding_model = "text-embedding-ada-002" 
        chroma_client = chromadb.EphemeralClient() # chroma DB client
        # dictionary containing text and embedded vector for each piece of text
        df = {'text': [], 'text_vector': []}
        # loop through lines and add values
        for line in text_transcript.split("\n"): 
            df['text'].append(line)
            df['text_vector'].append(self.openAI_client.embeddings.create(
                input = line, model=embedding_model).data[0].embedding)
        # create the to be used embedding function that will be used to query the chroma DB
        embedding_function = OpenAIEmbeddingFunction(api_key=self.api_key, model_name=embedding_model)
        # create the chroma collection that will store the embedded transcript
        transcript_collection = chroma_client.create_collection(name='transcript_content', embedding_function=embedding_function)
        # convert dictionary to dataframe
        df = pd.DataFrame(data=df)
        df.drop(len(df)-1, inplace=True)
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
        return df.tail(3)
        