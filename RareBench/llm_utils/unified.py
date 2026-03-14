# llm_utils/unified.py
import os
import time
from typing import Optional, Literal, Dict, Any

Provider = Literal["openai", "anthropic", "gemini"]

class BaseLLMHandler:
    """Minimal, provider-agnostic surface your pipeline expects."""
    provider: Provider
    model_name: str
    total_tokens: int

    def get_completion(self, system_prompt: str, prompt: str, seed: Optional[int] = 42) -> Optional[str]:
        raise NotImplementedError

# ----------------------------- OpenAI / Azure -----------------------------
try:
    from openai import AzureOpenAI
    _HAS_AZURE = True
except Exception:
    _HAS_AZURE = False

import openai as _openai  # still used for non-Azure

class OpenAIChatHandler(BaseLLMHandler):
    """
    Supports:
      - Azure OpenAI (preferred if AZURE_OPENAI_ENDPOINT set)
      - OpenAI.com fallback
    Model resolution:
      - If Azure: `self.model` = deployment name (AZURE_OPENAI_DEPLOYMENT or EVAL)
      - Else:     `self.model` = raw OpenAI model id (e.g., gpt-4o, gpt-4o-mini, etc.)
    """
    def __init__(self, model: Optional[str] = None, role: str = "reasoner") -> None:
        self.provider = "openai"
        self.total_tokens = 0

        if os.getenv("AZURE_OPENAI_ENDPOINT") and _HAS_AZURE:
            self._is_azure = True
            self._client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            )
            if role in {"judge", "eval"}:
                self.model_name = model or os.getenv("AZURE_OPENAI_EVAL_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            else:
                self.model_name = model or os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_EVAL_DEPLOYMENT")
            if not self.model_name:
                raise RuntimeError("Missing Azure deployment: set AZURE_OPENAI_DEPLOYMENT (and/or AZURE_OPENAI_EVAL_DEPLOYMENT).")
        else:
            self._is_azure = False
            self._client = None
            self._configure_openai_key()
            # If `model` given, pass through; else pick a sane default
            self.model_name = model or "gpt-4o"

    def _configure_openai_key(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            # legacy file path fallback
            try:
                with open("llm_utils/gpt_key.txt", "r") as f:
                    key = f.readline().strip()
            except FileNotFoundError:
                pass
        if not key:
            raise RuntimeError("OpenAI key not found. Set OPENAI_API_KEY env or llm_utils/gpt_key.txt")
        _openai.api_key = key

    def get_completion(self, system_prompt: str, prompt: str, seed: Optional[int] = 42) -> Optional[str]:
        try:
            t0 = time.time()
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            if self._is_azure:
                resp = self._client.chat.completions.create(model=self.model_name, seed=seed, messages=msgs)
            else:
                resp = _openai.chat.completions.create(model=self.model_name, seed=seed, messages=msgs)
            msg = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            if usage:
                self.total_tokens += int(getattr(usage, "prompt_tokens", 0)) + int(getattr(usage, "completion_tokens", 0))
            print(f"[OpenAI] model={self.model_name} time={time.time()-t0:.2f}s")
            return msg
        except Exception as e:
            print("[OpenAI] error:", e)
            return None

# ----------------------------- Anthropic (Claude) -----------------------------
class AnthropicHandler(BaseLLMHandler):
    """
    Anthropic Messages API. Pass exact model id via `model` or CLAUDE_MODEL env.
    Example ids: "claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest", etc.
    """
    def __init__(self, model: Optional[str] = None) -> None:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.provider = "anthropic"
        self.model_name = model or os.getenv("CLAUDE_MODEL", "claude-4-5-sonnet")
        self.total_tokens = 0

    def get_completion(self, system_prompt: str, prompt: str, seed: Optional[int] = 42) -> Optional[str]:
        try:
            t0 = time.time()
            resp = self._client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.2,
            )
            text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
            # Usage is optional; Anthropic returns input/output tokens in `usage` on newer SDKs
            usage = getattr(resp, "usage", None)
            if usage and isinstance(usage, dict):
                self.total_tokens += int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
            print(f"[Anthropic] model={self.model_name} time={time.time()-t0:.2f}s")
            return text.strip()
        except Exception as e:
            print("[Anthropic] error:", e)
            return None

# ----------------------------- Google (Gemini) -----------------------------
class GeminiHandler(BaseLLMHandler):
    """
    Google Generative AI (Gemini). Pass exact model id via `model` or GEMINI_MODEL env.
    Examples (subject to your account access): "gemini-1.5-pro", "gemini-2.0-flash-exp", etc.
    """
    def __init__(self, model: Optional[str] = None) -> None:
        import google.generativeai as genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            # legacy file fallback
            try:
                with open("llm_utils/gemini_key.txt", "r") as f:
                    key = f.readline().strip()
            except FileNotFoundError:
                key = None
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY or create llm_utils/gemini_key.txt")
        genai.configure(api_key=key, transport="rest")
        self._genai = genai
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        self._model = genai.GenerativeModel(self.model_name)
        self.provider = "gemini"
        self.total_tokens = 0

    def get_completion(self, system_prompt: str, prompt: str, seed: Optional[int] = 42) -> Optional[str]:
        try:
            t0 = time.time()
            # Simple concat; your prompts already separate system/user meaningfully
            txt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{prompt}"
            resp = self._model.generate_content(txt)
            out = (getattr(resp, "text", None) or "").strip()
            # Gemini usage fields vary; skip token accounting if missing
            print(f"[Gemini] model={self.model_name} time={time.time()-t0:.2f}s")
            return out
        except Exception as e:
            print("[Gemini] error:", e)
            return None

# ----------------------------- Factory -----------------------------
def get_handler(provider: Provider, model: Optional[str] = None, role: str = "reasoner") -> BaseLLMHandler:
    if provider == "openai":
        return OpenAIChatHandler(model=model, role=role)
    elif provider == "anthropic":
        return AnthropicHandler(model=model)
    elif provider == "gemini":
        return GeminiHandler(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
