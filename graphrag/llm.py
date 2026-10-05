"""Optional free local language model (Google FLAN-T5) that writes the final answer.
It runs on the Colab CPU/GPU - no API key, no payment."""


def load_llm(model_name="google/flan-t5-base", max_new_tokens=140):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    def generate(prompt):
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=3)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    return generate
