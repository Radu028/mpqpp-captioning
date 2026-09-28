import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL = "WueNLP/centurio_qwen"
REVISION = "bf6617eb01171dba83391e0ff57bf03420721ba5"


def load(**kwargs):
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        **kwargs,
    )
    return model, processor


def chat(prompt, answer=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages
