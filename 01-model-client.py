from dotenv import load_dotenv
import os
import asyncio

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import UserMessage

load_dotenv()

async def main():
    aopenai_client = AzureOpenAIChatCompletionClient(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    result = await aopenai_client.create([
        UserMessage(
            content="What is AutoGen framework from Microsoft Research?",
            source="user",
        ),
    ])

    print(result.content)

if __name__ == "__main__":
    asyncio.run(main())