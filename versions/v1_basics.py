"""
Version 1 - LangChain basics (no retrieval, no vector database)

Project Title:      Health FAQ RAG Chatbot (Group 63)
Problem Statement:  Cancer information online is scattered and unreliable; general chatbots can
                    invent medical facts with no sources.
Proposed Solution:  A RAG chatbot grounded in NIH Cancer.gov content, built up from V1 to V4.
Target Users:       Patients, carers and the public with general cancer questions.
Data/Documents:     MedQuAD CancerGov subset (729 NIH Q-A pairs, 98 cancer topics).
Expected Output:    A plain-language answer with sources, or a refusal when out of scope.
(Full text: versions/common.py)

Flow:
    Input -> clean_input (RunnableLambda) -> {"question": RunnablePassthrough}
          -> ChatPromptTemplate -> ChatGroq (LLM) -> StrOutputParser
          -> PydanticOutputParser (falls back to plain text if the model skips the JSON format)
          -> HealthAnswer

The LLM answers from its own training knowledge only. That is the problem V2-V4 fix: V1 has
no way to ground its answer in NIH documents or show sources.

Run: python versions/v1_basics.py ["your question"]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from versions.common import get_llm


class HealthAnswer(BaseModel):
    answer: str = Field(description="A short plain-language answer, 2-4 sentences")
    key_points: list[str] = Field(description="2-4 key points as short bullet phrases")
    disclaimer: str = Field(description="One sentence advising the user to consult a doctor")


parser = PydanticOutputParser(pydantic_object=HealthAnswer)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful health-information assistant for general questions about cancer. "
     "Explain in plain language for a non-medical reader. If the question is not about health, "
     "say you can only help with health questions. Always reply in the JSON format below, "
     "even when declining.\n\n{format_instructions}"),
    ("human", "{question}"),
]).partial(format_instructions=parser.get_format_instructions())


def parse_answer(text: str) -> HealthAnswer:
    """Parse the JSON reply; if the model answered in plain text, keep its words as the answer."""
    try:
        return parser.parse(text)
    except OutputParserException:
        return HealthAnswer(
            answer=text.strip(),
            key_points=[],
            disclaimer="Please consult a medical professional for advice about your situation.",
        )


def build_chain():
    clean_input = RunnableLambda(lambda q: " ".join(q.split()), name="clean_input")
    return (
        clean_input
        | {"question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
        | RunnableLambda(parse_answer, name="pydantic_parser")
    ).with_config(run_name="V1 LangChain basics")


def ask(question: str, config: dict | None = None) -> HealthAnswer:
    config = dict(config or {})
    config["tags"] = ["v1", *config.get("tags", [])]
    return build_chain().invoke(question, config=config)


def main():
    questions = sys.argv[1:] or [
        "What are the early warning signs of melanoma?",
        "How is hairy cell leukemia treated?",
    ]
    for q in questions:
        result = ask(q)
        print(f"\nQ: {q}")
        print(f"A: {result.answer}")
        for point in result.key_points:
            print(f"   - {point}")
        print(f"   ({result.disclaimer})")


if __name__ == "__main__":
    main()
