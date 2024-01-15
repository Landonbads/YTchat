import gradio as gr
from openai import OpenAI
import tiktoken
import transcript_handler

class GradioApp:
    def __init__(self, openAI_key):
        self.api_key = openAI_key
        self.client = OpenAI(api_key = openAI_key)
        self.transcript = None
        self.embedded_transcript = None

    def load_video(self,url):
        self.transcript = transcript_handler.Transcript(url,self.client,self.api_key)
        self.transcript.get_transcript() # load transcript text
        # embed transcript into chroma vector DB
        self.embedded_transcript = self.transcript.embed_transcript(self.transcript.text_transcript)
        return "Video loaded successfully!"
        
    def ask_question(self,message,chat_history):
        assistant_context = """You are a youtube video assistant. 
        Given the provided youtube video snippets answer the user's questions.\n 
        snippet: """
        best_matches = self.transcript.query_collection(message, 30, self.embedded_transcript)
        for index, row in best_matches.iterrows():
            assistant_context += row['content'] + "\n"
        gpt_context = [{"role": "user", "content": assistant_context}]
        for human, assistant in chat_history:
            gpt_context.append({"role": "user", "content": human}) 
            gpt_context.append({"role": "assistant", "content": assistant})  
        gpt_context.append({"role": "user", "content": message})
        # Get response from OpenAI
        response = self.client.chat.completions.create(model="gpt-3.5-turbo",
                                                messages=gpt_context)
        chat_history.append((message, response.choices[0].message.content))
        return "", chat_history
    
    # separate function for getting the summary of entire video
    def get_summary(self):
        # count number of tokens in string to make sure <= 4k
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(self.transcript.text_transcript))
        if num_tokens > 4000:
            return "Video is too costly to perform summary!"
        gpt_context = f"context: {self.transcript.text_transcript}\n"
        question = [{"role": "user", "content": gpt_context + """This is the transcript of a youtube video 
        i'm watching. Please provide a brief summary of the video."""}]
        # get response from OpenAI
        response = self.client.chat.completions.create(model="gpt-3.5-turbo",
                                                messages=question)
        # return youtube summary
        return response.choices[0].message.content
    

    def build_app(self):
        with gr.Blocks(theme=gr.themes.Monochrome()) as app:
            gr.Markdown("## YTchat")
            with gr.Row():
                url_input = gr.Textbox(placeholder="Enter Video/Shorts URL")
                load_output = gr.Textbox(label="Load Status", interactive=False)
                load_button = gr.Button("Load Video")
            with gr.Column():
                summary_output = gr.Textbox(label="Summary", interactive=False)
                summary_button = gr.Button("Load video summary")
            with gr.Column():
                chatbot = gr.Chatbot()
                msg = gr.Textbox(placeholder="Ask a question about a specific part of the video...")
                clear = gr.ClearButton([msg, chatbot])
                msg.submit(self.ask_question, [msg,chatbot], [msg, chatbot])

            summary_button.click(fn=self.get_summary ,outputs=summary_output)
            load_button.click(fn=self.load_video, inputs=url_input, outputs=load_output)

        return app

    
