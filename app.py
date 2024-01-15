import os
import gradio_app

def main():
    app = gradio_app.GradioApp(os.getenv('OPENAI_KEY'))
    app.build_app().launch()
    
if __name__ == "__main__":
    main()