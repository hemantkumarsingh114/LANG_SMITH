from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# 1. Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

# 2. LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
r'''llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation")

llm = ChatHuggingFace(llm=llm_endpoint)'''

# 3. Parser
parser = StrOutputParser()

# 4. Chain: prompt → model → parser
chain = prompt | llm | parser

# 5. Run it
result = chain.invoke({"question": "What is the capital of france?"})
print(result)