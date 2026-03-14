# core/llm_handlers.py
import os
from typing import Optional, Literal

Provider = Literal["openai","anthropic","gemini"]

# OpenAI / Azure
from openai import AzureOpenAI
import openai as _openai

# Anthropic
import anthropic

# Gemini
import google.generativeai as genai

class BaseLLMHandler:
    provider: Provider
    model_name: str
    total_tokens: int

    def get_completion(self, system_prompt: str, prompt: str, seed: Optional[int] = 42) -> Optional[str]:
        raise NotImplementedError

# ---------------- OpenAI / Azure ----------------
class OpenAIChatHandler(BaseLLMHandler):
    def __init__(self, model: Optional[str] = None, role: str = "reasoner") -> None:
        self.provider = "openai"
        self.total_tokens = 0
        self._is_azure = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))
        if self._is_azure:
            self._client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            )
            self.model_name = model or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            if not self.model_name:
                raise RuntimeError("Set AZURE_OPENAI_DEPLOYMENT or pass --model.")
        else:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("Set OPENAI_API_KEY or Azure envs.")
            _openai.api_key = key
            self._client = None
            self.model_name = model or "gpt-4o-mini"

    def get_completion(self, system_prompt: str, prompt: str, seed: Optional[int] = 42) -> Optional[str]:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            if self._is_azure:
                # Azure o4-mini: DO NOT send temperature (unsupported → 400)
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    seed=seed,
                    messages=messages,
                    # no temperature here
                )
            else:
                # OpenAI.com: ok to send temperature; remove it if you want default behavior
                resp = _openai.chat.completions.create(
                    model=self.model_name,
                    seed=seed,
                    messages=messages,
                    # You can keep or drop temperature on public API.
                    # temperature=0.2,
                )

            return resp.choices[0].message.content or ""
        except Exception as e:
            print("[OpenAI] error:", e)
            return ""


# ---------------- Anthropic ----------------
class AnthropicHandler(BaseLLMHandler):
    def __init__(self, model: Optional[str] = None) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.provider = "anthropic"
        self.model_name = model or os.getenv("CLAUDE_MODEL","claude-4.5-sonnet")
        self.total_tokens = 0

    def get_completion(self, system_prompt: str, prompt: str, seed: Optional[int] = 42) -> Optional[str]:
        try:
            resp = self._client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[{"role":"user","content":prompt}],
                max_tokens=2048,
                temperature=0.2,
            )
            parts = []
            for block in resp.content:
                if getattr(block, "type", "") == "text":
                    parts.append(block.text)
            return "\n".join(parts).strip()
        except Exception as e:
            print("[Anthropic] error:", e)
            return ""

# ---------------- Gemini ----------------
class GeminiHandler(BaseLLMHandler):
    def __init__(self, model: Optional[str] = None) -> None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY")
        genai.configure(api_key=key, transport="rest")
        self.model_name = model or os.getenv("GEMINI_MODEL","models/gemini-2.5-pro")
        self._model = genai.GenerativeModel(self.model_name)
        self.provider = "gemini"
        self.total_tokens = 0

    def get_completion(self, system_prompt: str, prompt: str, seed: Optional[int] = 42) -> Optional[str]:
        try:
            txt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{prompt}"
            out = self._model.generate_content(txt)
            return (getattr(out, "text", None) or "").strip()
        except Exception as e:
            print("[Gemini] error:", e)
            return ""

def get_handler(provider: Provider, model: Optional[str] = None, role: str = "reasoner"):
    if provider == "openai":
        return OpenAIChatHandler(model=model, role=role)
    if provider == "anthropic":
        return AnthropicHandler(model=model)
    if provider == "gemini":
        return GeminiHandler(model=model)
    raise ValueError(f"Unknown provider: {provider}")
