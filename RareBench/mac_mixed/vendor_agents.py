# mac_mixed/vendor_agents.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from mac_mixed.vendor_clients.openai_azure import call_openai_azure
from mac_mixed.vendor_clients.gemini import call_gemini
from mac_mixed.vendor_clients.claude import call_claude

@dataclass
class BaseVendorAgent:
    name: str
    system_message: str
    temperature: float = 1.0

    def _history_to_strings(self, chat_history: List[Dict[str, Any]]) -> List[str]:
        # Convert MAC chat_history (dicts with 'name', 'content') into linear text turns
        msgs = []
        for m in chat_history:
            nm = m.get("name", "Unknown")
            ct = m.get("content", "")
            msgs.append(f"{nm}: {ct}")
        return msgs

    def generate_reply(self, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIAzureAgent(BaseVendorAgent):
    deployment: Optional[str] = None
    def generate_reply(self, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = call_openai_azure(self.system_message, self._history_to_strings(chat_history),
                                 deployment=self.deployment, temperature=self.temperature)
        return {"role": "assistant", "name": self.name, "content": text}

class GeminiAgent(BaseVendorAgent):
    model: Optional[str] = None
    def generate_reply(self, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = call_gemini(self.system_message, self._history_to_strings(chat_history),
                           model=self.model, temperature=self.temperature)
        return {"role": "assistant", "name": self.name, "content": text}

class ClaudeAgent(BaseVendorAgent):
    model: Optional[str] = None
    def generate_reply(self, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = call_claude(self.system_message, self._history_to_strings(chat_history),
                           model=self.model, temperature=self.temperature)
        return {"role": "assistant", "name": self.name, "content": text}
