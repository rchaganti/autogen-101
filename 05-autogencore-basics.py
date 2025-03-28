from dataclasses import dataclass
from typing import Callable
import asyncio
from sympy import isprime
import random

from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
from autogen_core import AgentId, SingleThreadedAgentRuntime

@dataclass
class Message:
    content: int

@default_subscription
class Generator(RoutedAgent):
    def __init__(self, generate_val: Callable[[], int]) -> None:
        super().__init__("A generator agent.")
        self._generate_val = generate_val

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        val = self._generate_val()
        print(f"{'-'*80}\nGenerator:\nWe got {message.content} prime numbers. \nNext number is {val}")
        await self.publish_message(Message(content=val), DefaultTopicId())  # type: ignore


@default_subscription
class Checker(RoutedAgent):
    def __init__(self, run_until: Callable[[int], bool]) -> None:
        super().__init__("A checker agent.")
        self.count = 0
        self._run_until = run_until
        self.prime_numbers = []

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        if message.content not in self.prime_numbers:
            if isprime(message.content):            
                self.count = self.count+1
                self.prime_numbers.append(message.content)
                if not self._run_until(self.count):
                    print(f"{'-'*80}\nChecker:\n{message.content} is a prime number, continue to next.")
                    await self.publish_message(Message(content=self.count), DefaultTopicId())
                else:                
                    print(f"{'-'*80}\nChecker:\nWe got {self.count} prime numbers, stopping.")
                    print(f"{'-'*80}\nPrime numbers generated: {self.prime_numbers}")
            else:
                print(f"{'-'*80}\nChecker:\n{message.content} is not a prime number, generate next.")
                await self.publish_message(Message(content=self.count), DefaultTopicId())
        else:
            print(f"{'-'*80}\nChecker:\n{message.content} exists in the generated prime numbers, continue to next.")
            await self.publish_message(Message(content=self.count), DefaultTopicId())

async def main():
    runtime = SingleThreadedAgentRuntime()

    await Generator.register(
        runtime,
        "generator",
        lambda: Generator(generate_val=lambda: random.randint(1, 1000)),
    )

    await Checker.register(
        runtime,
        "checker",
        lambda: Checker(run_until=lambda count: count >= 10),
    )

    runtime.start()
    await runtime.send_message(Message(1), AgentId("checker", "default"))
    await runtime.stop_when_idle()

asyncio.run(main())