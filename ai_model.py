from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI


class AIModel(ABC):
    @abstractmethod
    def set_system_prompt(self, system_prompt: str):
        pass

    @abstractmethod
    def set_api_key(self):
        pass

    @abstractmethod
    def set_llm(self, temperature: float = 0.5):
        pass

class AlbertEinstein(AIModel):
    def __init__(self, model):
        self.name = "Albert Einstein"
        self.model = model
        self.api_key = None
        self.system_prompt = None
        self.temperature = 0.5
        self.llm = None
        self.chain = None

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

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