from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

import asyncio
from dotenv import load_dotenv
import os
import json
import requests
from datetime import datetime

from typing import Any
from autogen_core.models import ModelInfo

load_dotenv()

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

aoi_client = AzureOpenAIChatCompletionClient(
    model=model,
    api_key=api_key,
    endpoint=endpoint,
    api_version=api_version,
    model_info=model_info,
)

def get_weather(city: str, date: datetime = None) -> dict[str, Any]:
    """
    Get the weather at a given location on a given date or current weather.

    Args:
        city: The city name, e.g. Bengaluru.
        date: Date on which the weather at the given location should be determined. This defaults to the current weather when a date is not specified.

    Returns:
        JSON string with the city name, date, and temperature.
    """
    api_key = os.getenv("VISUAL_CROSSING_API_KEY")
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    request_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{date}?unitGroup=metric&key={api_key}&contentType=json"
    response = requests.get(request_url)

    if response.status_code != 200:
        return json.dumps({
            "error": "Invalid city name or date"
        })
    else:
        respJson = response.json()
        return json.dumps({
            "city": city,
            "date": date,
            "temperature": respJson["days"][0]["temp"]
        })

agent = AssistantAgent(
    name="weather_agent",
    model_client=aoi_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,
)

async def main() -> None:
    await Console(
        agent.run_stream(
            task="What is the weather in London?"
        )
    )

asyncio.run(main())
