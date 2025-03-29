import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import random
from sympy import isprime

import os
from dotenv import load_dotenv

load_dotenv(override=True)

def generate_number(start: int, end: int) -> int:
    """Generate a random number."""
    return random.randint(start,end)

def check_is_prime_number(number: int) -> bool:
    """check if the random number is a prime."""
    return isprime(number)

async def main():
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

    generator_agent = AssistantAgent(
        "generator",
        description="An agent that generates random numbers.",
        model_client=model_client,
        tools=[generate_number],
        system_message="""
        You are a numbers wizard. Use the tools to generate random number.
        Maintain a list of generated numbers. If a new number is generated, check if it is already in the list.
        If it is, generate a new number. If it is not, add it to the list and return the number.
        """,
    )

    verifier_agent = AssistantAgent(
        "verifier",
        description="An agent that verifies if a number is prime.",
        model_client=model_client,
        tools=[check_is_prime_number],
        system_message=""""
            You are a powerful calculator, use the tools to verify if a number is prime number.
            Skip already verified numbers.
            Respond with DONE when all required prime numbers are found and return all prime numbers you found."
        """
    )

    text_termination = TextMentionTermination("DONE")

    team = RoundRobinGroupChat([generator_agent, verifier_agent], termination_condition=text_termination)

    await team.reset()
    await Console(team.run_stream(task="Find 5 random prime numbers between 1 and 15."))

result = asyncio.run(main())
print(result)