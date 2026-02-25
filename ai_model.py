from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI


class AIModel(ABC):
    @abstractmethod
    def set_api_key(self):
        pass

    @abstractmethod
    def set_llm(self, temperature: float = 0.5):
        pass

class AlbertEinstein(AIModel):
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

    def __init__(self, model):
        self.name = "Albert Einstein"
        self.model = model
        self.api_key = None
        self.system_prompt = AlbertEinstein.system_prompt
        self.temperature = 0.5
        self.llm = None
        self.chain = None

    def set_api_key(self, environment_variable: str = "GEMINI_API_KEY"):
        load_dotenv()
        self.api_key = os.getenv(environment_variable)

    def set_llm(self, temperature: float = 0.5):
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            temperature=0.5
        )

    def set_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            (MessagesPlaceholder(variable_name="history")),
            ("user", "{input}")]
        )

        self.chain = prompt | self.llm | StrOutputParser()