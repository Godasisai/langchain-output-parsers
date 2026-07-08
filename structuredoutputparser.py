from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


class Facts(BaseModel):
    fact_1: str = Field(description="Fact 1 about the topic")
    fact_2: str = Field(description="Fact 2 about the topic")
    fact_3: str = Field(description="Fact 3 about the topic")


structured_model = model.with_structured_output(Facts)

result = structured_model.invoke(
    "Give 3 facts about black holes."
)

print(result)
print(result.fact_1)
print(result.fact_2)
print(result.fact_3)