# mac_mixed/vendor_clients/openai_azure.py
import os
from openai import AzureOpenAI

def call_openai_azure(system_message: str, messages: list[str], deployment: str = None, temperature: float = 1.0, timeout: int = 180):
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        timeout=timeout,
    )
    model = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-1120")
    chat = [{"role": "system", "content": system_message}]
    for m in messages:
        # We treat the entire prior message as 'user' content for simplicity
        chat.append({"role": "user", "content": m})
    resp = client.chat.completions.create(
        model=model,
        messages=chat,
        temperature=float(temperature),
    )
    return resp.choices[0].message.content.strip()
