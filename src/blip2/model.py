import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor

MODEL = "Salesforce/blip2-opt-6.7b-coco"
REVISION = "fa20c535dcf60c0fab06d95c88729b9aa482b34b"


def load(dtype):
    processor = Blip2Processor.from_pretrained(MODEL, revision=REVISION)
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, dtype=dtype)
    # the processor inserts the query tokens as <image> tokens, both sides must agree on id and count
    model.config.image_token_index = processor.tokenizer.convert_tokens_to_ids(str(processor.image_token))
    processor.num_query_tokens = model.config.num_query_tokens
    return model.to("cuda").eval(), processor


def to_cuda(batch, dtype):
    return {k: v.to("cuda", dtype) if v.is_floating_point() else v.to("cuda") for k, v in batch.items()}


def caption(model, processor, image):
    inputs = to_cuda(processor(images=image, return_tensors="pt"), model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False, num_beams=1)
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()
