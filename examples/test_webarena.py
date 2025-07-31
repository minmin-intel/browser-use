from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from browser_use import Agent
from pydantic import SecretStr
from dotenv import load_dotenv
import os
import json

# import sys
# print(f"path: {sys.path}")

def sample_tasks(domain="shopping_admin"):
    import pandas as pd
    WORKDIR= os.environ.get("WORKDIR", os.getcwd())
    DATAPATH= os.path.join(WORKDIR, "datasets", "webarena", "webarena_raw.json")
    with open(DATAPATH, "r") as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier manipulation
    task_id_list = []
    intent_list = []
    ref_answer_list = []
    intent_template_id_list = []
    for task in data:
        if task["sites"][0] in [domain]:
            task_id = task["task_id"]
            intent= task["intent"]
            ref_answer = task['eval']["reference_answers"]
            intent_template_id = task["intent_template_id"]
            if ref_answer:
                task_id_list.append(task_id)
                intent_list.append(intent)
                ref_answer_list.append(ref_answer)
                intent_template_id_list.append(intent_template_id)

    df = pd.DataFrame({
        "task_id": task_id_list,
        "intent": intent_list,
        "reference_answer": ref_answer_list,
        "intent_template_id": intent_template_id_list
    })

    # group by intent_template_id and sample 1 task from each group
    # use random seed for reproducibility
    df = df.groupby("intent_template_id").apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
    if df.shape[0] > 30:
       df = df.sample(n=30, random_state=42, replace=False, axis=0, ignore_index=True)
    df.to_csv(os.path.join(WORKDIR, "datasets", "webarena", f"{domain}_tasks.csv"), index=False)
    return df

def augment_prompt_for_shopping_admin(intent, url):
    """
    Augment the prompt with specific instructions for the shopping admin domain.
    """
    return f"""
    Navigate to {url} and login with the credentials:
    username: admin
    password: admin1234
    Then, {intent}
    """


def setup_agent(args, prompt):
    load_dotenv()

    if args.model_provider == "deepseek":
        print("Using DeepSeek model...")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        # Initialize the model
        llm=ChatDeepSeek(
            base_url='https://api.deepseek.com/v1', 
            model='deepseek-chat', 
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

    return agent
    


if __name__ == "__main__":
    import asyncio
    import argparse
    parser = argparse.ArgumentParser(description="bechmark browser-use agent on webarena.")
    parser.add_argument("--model_provider", type=str, default="deepseek", help="Model provider to use (default: deepseek)")
    args = parser.parse_args()

    df = sample_tasks()
    # df = df.head(2)
    print(f"Total tasks: {df.shape[0]}")

    for index, row in df.iterrows():
        intent = row["intent"]
        print(f"Running task {index + 1}/{df.shape[0]}: {intent}")
        url = "http://localhost:8083/admin"
        prompt = augment_prompt_for_shopping_admin(intent, url)
        agent = setup_agent(args, prompt)
        asyncio.run(agent.run(max_steps=30))
        print("==== Task Completed ====" * 50)