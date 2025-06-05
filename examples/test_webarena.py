from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from browser_use import Agent
from pydantic import SecretStr
from dotenv import load_dotenv
import os

# import sys
# print(f"path: {sys.path}")



async def main(args):
    load_dotenv()
    prompt = """Nagivate to http://localhost:9083/admin and login with the credentials:
        username: admin
        password: admin1234
        Then, find out What are the top-3 best-selling product in **Jan 2023**? Today is May 2025.
"""


    if args.model_provider == "deepseek":
        print("Using DeepSeek model...")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        # Initialize the model
        llm=ChatDeepSeek(
            base_url='https://api.deepseek.com/v1', 
            model='deepseek-reasoner', 
            api_key=SecretStr(api_key),
            temperature=0.3,
            )
        # Create agent with the model
        agent = Agent(
            task=prompt,
            llm=llm,
            use_vision=False,
            enable_memory=False,
        )
    elif args.model_provider == "together":
        print("Using Together model API...")
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable is not set. Please set it to use Together model.")
        llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            # model = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            model = "Qwen/Qwen2.5-VL-72B-Instruct",
            api_key=SecretStr(os.getenv("TOGETHER_API_KEY")),
            temperature=0.3,
        )
        agent = Agent(
            task=prompt,
            llm=llm,
            use_vision=True,
            enable_memory=False,
        )
    else:
        raise ValueError(f"Unsupported model provider: {args.model_provider}. Supported providers are: deepseek, together.")

    history = await agent.run(max_steps=10)

    print(f"URLs: {history.urls()}")              # List of visited URLs
    print(f"Actions: {history.action_names()}")     # Names of executed actions
    # history.extracted_content() # Content extracted during execution
    # print(f"Errors: {history.errors()}")           # Any errors that occurred
    # print(f"Actions with params: {history.model_actions()}")     # All actions with their parameters
    # print(f"Screenshots: {history.screenshots()}")       # List of screenshot paths
    


if __name__ == "__main__":
    import asyncio
    import argparse
    parser = argparse.ArgumentParser(description="bechmark browser-use agent on webarena.")
    parser.add_argument("--model_provider", type=str, default="deepseek", help="Model provider to use (default: deepseek)")
    args = parser.parse_args()
    asyncio.run(main(args))