import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Set Project Name:
os.environ['LANGCHAIN_PROJECT'] = 'Sequental LLM App'

# 1. Prompt
prompt1 = PromptTemplate(
    template='Generate a brief report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# 2. LLM
llm1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
    temperature=0.5)
    
llm2 = ChatHuggingFace(llm=llm_endpoint)

# 3. Parser
parser = StrOutputParser()

# 4. Chain: prompt → model → parser
chain = prompt1 | llm2 | parser | prompt2 | llm1 | parser

config = {
    'run_name': 'sequential chain',
    'tags': ['llm app', 'report generation', 'summarization'],
    'metadata': {'model1': 'gemini-2.5-flash-lite', 'model1_temp': 0.7, 'parser': 'StrOutputParser'}
}

# 5. Run it
result = chain.invoke({'topic': 'Unemployment in India'}, config=config)
print(result)