import os
import gradio as gr

from langchain_core.messages import HumanMessage, AIMessage

from ai_model import AlbertEinstein

class ChatBot:
    def __init__(self, ai_model: AlbertEinstein,
                 user_image=None,
                 avatar_image=None,
                 theme=gr.themes.Soft()):
        self.theme = theme
        self.ai_model = ai_model
        self.user_image = user_image
        self.avatar_image = avatar_image

    def chat(self, user_in, hist):
        langchain_history = []
        for item in hist:
            if item["role"] == "user":
                langchain_history.append(HumanMessage(content=item["content"]))
            elif item["role"] == "assistant":
                langchain_history.append(AIMessage(content=item["content"]))

        response = self.ai_model.chain.invoke({"input": user_in,
                                               "history": langchain_history})

        return "", hist + [
            {"role": "user", "content": user_in},
            {"role": "assistant", "content": response},
        ]

    def run(self):
        page = gr.Blocks(
            title=f"Chat with {self.ai_model.name}"
        )

        with page:
            gr.Markdown(
                f"""
                # Chat with {self.ai_model.name}!\n
                Welcome to your personal conversation with {self.ai_model.name}.
                """
            )

            chatbot = gr.Chatbot(show_label=False,
                                 elem_id="chatbot",
                                 avatar_images=[self.user_image, self.avatar_image])
            msg = gr.Textbox(show_label=False,
                            placeholder=f"Ask {self.ai_model.name} anything...",)


            msg.submit(self.chat,
                       inputs=[msg, chatbot],
                       outputs=[msg, chatbot])

            clear = gr.Button("Clear Chat")
            clear.click(fn=lambda: chatbot.clear(), inputs=[], outputs=[chatbot])

        page.launch(
            theme=self.theme
        )

if __name__ == "__main__":

    einstein_chatbot = AlbertEinstein('gemini-2.5-flash')
    einstein_chatbot.set_system_prompt(system_prompt ="""
        You are Albert Einstein.
        Answer questions through Einstein's questioning and reasoning...
        You will speak from your point of view. You will share personal things from your life
        even when the user doesn't ask for it. For example, if the user asks about the theory
        of relativity, you will share your personal experiences with it and not only
        explain the theory.
        Answer in 2-6 sentences.
        You should have a sense of humor.
        """)

    einstein_chatbot.set_api_key("GEMINI_API_KEY")
    einstein_chatbot.set_llm()
    einstein_chatbot.set_chain()

    chat_bot = ChatBot(einstein_chatbot,
                       avatar_image='resources/einstein.png',
                       theme=gr.themes.Glass())
    chat_bot.run()

