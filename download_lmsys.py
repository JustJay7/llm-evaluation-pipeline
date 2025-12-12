from datasets import load_dataset

ds = load_dataset("lmsys/lmsys-chat-1m", use_auth_token=True)

print(ds)
