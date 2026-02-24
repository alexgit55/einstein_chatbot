from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI



if __name__ == "__main__":
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.5
    )

    response=llm.invoke([{
        "role": "user",
        "content": "Hi there, how are you?"
    }])

    print(response)

    # print("Hi, I am Albert, how can I help you today?")
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() == "exit":
    #         break
    #     print(f"Cool, thanks for sharing that { user_input}")
