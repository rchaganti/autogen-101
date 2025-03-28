from dotenv import load_dotenv
import os
import asyncio

from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import UserMessage

load_dotenv()

async def main():
    ollama_client = OllamaChatCompletionClient(
        model="llama3.2:3b"
    )

    result = await ollama_client.create([
        UserMessage(
            content="What is AutoGen framework from Microsoft Research?",
            source="user",
        ),
    ])

    print(result.content)

if __name__ == "__main__":
    asyncio.run(main())