from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo

from dotenv import load_dotenv
import os

import asyncio
import random
from sympy import isprime

load_dotenv(override=True)

api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

model_info = ModelInfo(
    vision=False,
    structured_output=False,
    function_calling=True,
    streaming=False,
    json_output=False,
    family="gpt-4o",
)

model_client = AzureOpenAIChatCompletionClient(
    model=model,
    api_key=api_key,
    endpoint=endpoint,
    api_version=api_version,
    model_info=model_info,
)

def generate_number(start: int, end: int) -> int:
    """Generate a random number."""
    return random.randint(start,end)

def check_is_prime_number(number: int) -> bool:
    """check if the random number is a prime."""
    return isprime(number)

async def main():
    prime_number_assistant = AssistantAgent(
        "prime_number_assistant",
        model_client=model_client,
        tools=[generate_number, check_is_prime_number],
        system_message=""""
            You are a helpful AI assistant, use the tools to generate random number and check if the number is prime or not.
            Skip already generated or verified numbers.
            Respond with DONE when all required prime numbers are found and return all prime numbers you found."
        """
    )

    termination_condition = TextMentionTermination("DONE")

    team = RoundRobinGroupChat(
        [prime_number_assistant],
        termination_condition=termination_condition,
    )

    await team.reset()
    await Console(team.run_stream(task="Find 5 random prime numbers between 1 and 15."))

asyncio.run(main())