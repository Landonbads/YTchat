import os
import gradio_app
from dotenv import load_dotenv

def main():
    load_dotenv()
    app = gradio_app.GradioApp(os.getenv('OPENAI_KEY'))

    port = int(os.getenv("PORT", 7860))

    app.build_app().launch(
        server_name="0.0.0.0",
        server_port=port,
    )

if __name__ == "__main__":
    main()
