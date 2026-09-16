import os
import requests
from langsmith import Client
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Set Project Name:
os.environ['LANGCHAIN_PROJECT'] = 'ReAct Agent'

# 1: Search Tool and Weather Tool
search_tool = DuckDuckGoSearchRun()

# Weather API
@tool
def get_weather_data(city: str) -> dict:
    """This function fetches the current weather data for a given city"""

    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    return response.json()

# 2: LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
'''llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",  # Qwen/Qwen2.5-7B-Instruct
    task="text-generation")
    
llm = ChatHuggingFace(llm=llm_endpoint)'''

# 3: Pull the React prompt from LangSmith
client = Client()
prompt = client.pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)

# 4: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# 5: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon?
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# 6: Invoke
response = agent_executor.invoke({"input": "Identify the birthplace city of Kalpana Chawla (search) and give its current temperature."})
print(response)

print(response['output'])