import os
import gradio_app
from dotenv import load_dotenv


def main():
    load_dotenv()
    app = gradio_app.GradioApp(os.getenv('OPENAI_KEY'))
    app.build_app().launch()
    
if __name__ == "__main__":
    main()