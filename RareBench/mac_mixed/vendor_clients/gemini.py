# mac_mixed/vendor_clients/gemini.py
import os
import google.generativeai as genai

def call_gemini(system_message: str, messages: list[str], model: str = None, temperature: float = 1.0):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    mdl = model or os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
    sys = [{"role": "user", "parts": [{"text": f"[SYSTEM]\n{system_message}"}]}]
    convo = sys + [{"role": "user", "parts": [{"text": m}]} for m in messages]
    # Gemini uses a single prompt with history or a chat object; here we feed concatenated history
    prompt = "\n\n---\n\n".join([p["parts"][0]["text"] for p in convo])
    gen = genai.GenerativeModel(mdl)
    out = gen.generate_content(prompt, generation_config={"temperature": float(temperature)})
    return (out.text or "").strip()
