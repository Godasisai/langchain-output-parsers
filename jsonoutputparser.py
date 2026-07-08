from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# JSON Parser
parser = JsonOutputParser()

# Prompt
prompt = PromptTemplate(
    template="""
Generate information about {topic}.

{format_instructions}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Chain
chain = prompt | llm | parser

# Invoke
result = chain.invoke({"topic": "Black Hole"})

print(result)
print(type(result))