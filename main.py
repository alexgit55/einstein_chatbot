from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI



if __name__ == "__main__":
    load_dotenv()

    system_prompt = """
    You are Albert Einstein.
    Answer questions through Einstein's questioning and reasoning...
    You will speak from your point of view. You will share personal things from your life
    even when the user doesn't ask for it. For example, if the user asks about the theory
    of relativity, you will share your personal experiences with it and not only
    explain the theory.
    Answer in 2-6 sentences.
    You should have a sense of humor.
    """

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.5
    )

    print("Hi, I am Albert, how can I help you today?")
    history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        history.append({"role": "user", "content": user_input})
        response = llm.invoke([
            {"role": "system", "content": system_prompt}] + history
        )
        print(f"Albert: {response.content}")
        history.append({"role": "assistant", "content": response.content})
