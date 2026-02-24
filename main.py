from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (MessagesPlaceholder(variable_name="history")),
        ("user", "{input}")]
    )

    chain = prompt | llm | StrOutputParser()

    print("Hi, I am Albert, how can I help you today?")
    history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chain.invoke({"input": user_input, "history": history})
        print(f"Albert: {response}")
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))
