from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.ui import Console

from dotenv import load_dotenv
import os

import asyncio
import random
from sympy import isprime

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
model_client = AzureOpenAIChatCompletionClient(
    model=model,
    api_key=api_key,
    endpoint=endpoint,
    api_version=api_version,
)

count = 0

def generate_number(start: int, end: int) -> int:
    """Generate a random number."""
    return random.randint(start,end)

def check_is_prime_number(number: int) -> bool:
    """check if the random number is a prime."""
    return isprime(number)

def increment_prime_number_count() -> str:
    """Increment number of prime numbers found and return a string"""
    global count
    count += 1
    return f"found {count} random prime numbers"

async def main():
    prime_number_assistant = AssistantAgent(
        "prime_number_assistant",
        model_client=model_client,
        tools=[generate_number, check_is_prime_number,increment_prime_number_count],
        system_message="You are a helpful AI assistant, use the tools to generate random number and check if the number is prime or not.",
    )

    termination_condition = TextMentionTermination("found 5 random prime numbers")

    team = RoundRobinGroupChat(
        [prime_number_assistant],
        termination_condition=termination_condition,
    )

    async for message in team.run_stream(task="Find 5 random prime numbers between 1 and 15. Skip already generated or verified numbers."):
        print(message)

asyncio.run(main())