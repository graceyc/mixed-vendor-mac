# llm_utils/api.py

import os
import time

# Try Azure client; keep OpenAI fallback
try:
    from openai import AzureOpenAI
    _HAS_AZURE = True
except Exception:
    _HAS_AZURE = False

import openai  # still needed for non-Azure fallback


class Openai_api_handler:
    """
    Unified OpenAI handler that supports:
      - Azure OpenAI via env vars (preferred)
      - OpenAI.com fallback via llm_utils/gpt_key.txt

    The constructor receives a *logical* label:
      - "reasoner" => use AZURE_OPENAI_DEPLOYMENT
      - "judge" / "eval" / "gpt4" => use AZURE_OPENAI_EVAL_DEPLOYMENT (fallback to DEPLOYMENT if unset)
      - For OpenAI.com fallback, "gpt4"/"chatgpt"/"chatgpt_instruct" map to public model IDs.
    """

    def __init__(self, model) -> None:
        self.model_label = (model or "").lower()

        # token counters (keep legacy fields for compatibility)
        self.total_tokens = 0
        self.gpt4_tokens = 0
        self.chatgpt_tokens = 0
        self.chatgpt_instruct_tokens = 0

        # Decide Azure vs OpenAI.com
        if os.getenv("AZURE_OPENAI_ENDPOINT") and _HAS_AZURE:
            # Azure path
            self._is_azure = True
            self._client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            )

            # Choose deployment by role/label
            eval_pref_labels = {"gpt4", "judge", "eval"}
            if self.model_label in eval_pref_labels:
                # Prefer EVAL deployment for judging; fallback to main if unset
                self.model = (
                    os.getenv("AZURE_OPENAI_EVAL_DEPLOYMENT")
                    or os.getenv("AZURE_OPENAI_DEPLOYMENT")
                )
            else:
                # Default to the main reasoning deployment
                self.model = (
                    os.getenv("AZURE_OPENAI_DEPLOYMENT")
                    or os.getenv("AZURE_OPENAI_EVAL_DEPLOYMENT")
                )

            if not self.model:
                raise RuntimeError(
                    "Set AZURE_OPENAI_DEPLOYMENT (and optionally AZURE_OPENAI_EVAL_DEPLOYMENT)."
                )

        else:
            # OpenAI.com fallback (reads key from file)
            self._is_azure = False
            try:
                with open("llm_utils/gpt_key.txt", "r") as f:
                    openai.api_key = f.readline().strip()
            except FileNotFoundError:
                raise RuntimeError(
                    "Missing llm_utils/gpt_key.txt and no Azure env vars found. "
                    "Either set Azure env vars or create llm_utils/gpt_key.txt with your OpenAI key."
                )

            # Map logical label -> public OpenAI model id
            if self.model_label in {"gpt4", "judge", "eval"}:
                # modern GPT-4 class model for judging
                self.model = "gpt-4o-2024-08-06"
            elif self.model_label in {"chatgpt", "chatgpt_instruct", "reasoner"}:
                self.model = "gpt-4o-mini-2024-07-18"
            else:
                self.model = "gpt-4o-mini-2024-07-18"

    def get_completion(self, system_prompt, prompt, seed=42):
        """
        Chat completion for both Azure and OpenAI.com.
        Returns string content or None on error.
        """
        try:
            t0 = time.time()
            if self._is_azure:
                completion = self._client.chat.completions.create(
                    model=self.model,  # Azure: this is the *deployment name*
                    seed=seed,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
            else:
                completion = openai.chat.completions.create(
                    model=self.model,
                    seed=seed,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )

            answer = completion.choices[0].message.content or ""
            usage = getattr(completion, "usage", None)
            if usage:
                in_tok = int(getattr(usage, "prompt_tokens", 0))
                out_tok = int(getattr(usage, "completion_tokens", 0))
                self.total_tokens += in_tok + out_tok
                # keep legacy per-label counters if you still look at them elsewhere
                if self.model_label == "gpt4":
                    self.gpt4_tokens += in_tok + out_tok
                elif self.model_label == "chatgpt":
                    self.chatgpt_tokens += in_tok + out_tok
                elif self.model_label == "chatgpt_instruct":
                    self.chatgpt_instruct_tokens += in_tok + out_tok
                print("Input tokens:", in_tok, "Output tokens:", out_tok)
            print(f"OpenAI API time: {time.time() - t0:.2f}s")
            print(f"[OpenAI call] Provider={'Azure' if self._is_azure else 'OpenAI.com'} | model={self.model} | label={self.model_label}")

            return answer
        except Exception as e:
            print(e)
            return None

    def get_embedding(self, text, model="text-embedding-3-large"):
        """
        Optional: only needed if embeddings are used.
        Under Azure, set AZURE_OPENAI_EMBED_DEPLOYMENT (deployment name).
        """
        text = (text or "").replace("\n", " ")
        try:
            if self._is_azure:
                emb_deploy = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", model)
                resp = self._client.embeddings.create(input=[text], model=emb_deploy)
            else:
                resp = openai.embeddings.create(input=[text], model=model)
            return {
                "text": text,
                "model": model,
                "embedding": resp.data[0].embedding,
                "usage": {
                    "input_tokens": int(getattr(resp, "usage", {}).get("prompt_tokens", 0)),
                    "total_tokens": int(getattr(resp, "usage", {}).get("total_tokens", 0)),
                },
            }
        except Exception as e:
            print(e)
            return None


class Zhipuai_api_handler:
    """
    Lazy-import ZhipuAI (GLM) so the package is not required unless you actually use it.
    Reads API key from llm_utils/glm_key.txt.
    """
    def __init__(self, model) -> None:
        import zhipuai  # lazy import
        self._zhipuai = zhipuai
        with open("llm_utils/glm_key.txt", "r") as f:
            self._zhipuai.api_key = f.readline().strip()

        if model == "glm4":
            self.model = "glm-4"
        elif model == "glm3_turbo":
            self.model = "glm-3-turbo"
        else:
            self.model = "glm-3-turbo"
        self.model_name = model

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            t0 = time.time()
            response = self._zhipuai.model_api.sse_invoke(
                model=self.model,
                prompt=system_prompt + prompt,
                temperature=0.15,
                top_p=0.7,
            )
            result = ""
            for event in response.events():
                if event.event == "add":
                    result += event.data
            print(f"{self.model} API time: {time.time() - t0:.2f}s")
            return result
        except Exception as e:
            print(e)
            return None


class Gemini_api_handler:
    """
    Lazy-import Google Generative AI so the package is not required unless you actually use it.
    Reads API key from llm_utils/gemini_key.txt.
    """
    def __init__(self, model) -> None:
        import google.generativeai as genai  # lazy import
        with open("llm_utils/gemini_key.txt", "r") as f:
            genai.configure(api_key=f.readline().strip(), transport="rest")

        if model == "gemini_pro":
            self.model_name = "gemini_pro"
            self._genai = genai
            self._model = genai.GenerativeModel("gemini-pro")
        else:
            self.model_name = model
            self._genai = genai
            self._model = genai.GenerativeModel("gemini-pro")

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            t0 = time.time()
            response = self._model.generate_content(system_prompt + prompt)
            result = getattr(response, "text", "") or ""
            print(f"Gemini API time: {time.time() - t0:.2f}s")
            return result
        except Exception as e:
            print(e)
            return None
