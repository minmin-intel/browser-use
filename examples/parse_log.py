import os

WORKDIR= os.environ.get("WORKDIR", os.getcwd())
DATAPATH= os.path.join(WORKDIR, "datasets", "webarena", "test_browser_use", "deepseek_chat_shop_admin.log")

def parse_log():
    with open(DATAPATH, "r") as f:
        lines = f.readlines()

    prompt_tokens = []
    completion_tokens = []
    cached_tokens = []
    for line in lines:
        if "*** Token usage:" in line:
            #*** Token usage: {'input_tokens': 10316, 'output_tokens': 180, 'total_tokens': 10496, 'input_token_details': {'cache_read': 8320}, 'output_token_details': {}}
            token_usage = eval(line.split(": ", 1)[1])
            prompt_tokens.append(token_usage['input_tokens'])
            completion_tokens.append(token_usage['output_tokens'])
            cached_tokens.append(token_usage['input_token_details']['cache_read'])

    # plot the token usage histograms
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        if i == 0:
            plt.hist(prompt_tokens, bins=50, color='blue', alpha=0.7, label='Prompt Tokens')
            plt.title('Prompt Tokens Distribution')
        elif i == 1:
            plt.hist(completion_tokens, bins=50, color='green', alpha=0.7, label='Completion Tokens')
            plt.title('Completion Tokens Distribution')
        else:
            plt.hist(cached_tokens, bins=50, color='orange', alpha=0.7, label='Cached Tokens')
            plt.title('Cached Tokens Distribution')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Frequency')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(WORKDIR, "datasets", "webarena", "token_usage_histograms.png"))
    # plt.show()


    # calculate median and max of each list
    prompt_tokens = {
        "median": sorted(prompt_tokens)[len(prompt_tokens) // 2],
        "max": max(prompt_tokens)
    }
    completion_tokens = {
        "median": sorted(completion_tokens)[len(completion_tokens) // 2],
        "max": max(completion_tokens)
    }
    cached_tokens = {
        "median": sorted(cached_tokens)[len(cached_tokens) // 2],
        "max": max(cached_tokens)
    }

    # print the results
    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Completion Tokens: {completion_tokens}")
    print(f"Cached Tokens: {cached_tokens}")

if __name__ == "__main__":
    parse_log()