import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")
dataset_id = "fka/awesome-chatgpt-prompts"

headers = {"Authorization": f"Bearer {token}"}
API_URL = f"https://datasets-server.huggingface.co/is-valid?dataset={dataset_id}"

response = requests.get(API_URL, headers=headers)

if response.ok:
    data = response.json()
    print(f"Dataset: {dataset_id}")
    print(f"Preview:    {'✅' if data.get('preview') else '❌'}")
    print(f"Viewer:     {'✅' if data.get('viewer') else '❌'}")
    print(f"Search:     {'✅' if data.get('search') else '❌'}")
    print(f"Filter:     {'✅' if data.get('filter') else '❌'}")
    print(f"Statistics: {'✅' if data.get('statistics') else '❌'}")
else:
    print(f"Error: {response.status_code} - {response.text}")
