"""fp32 PyTorch -> ONNX fp32 -> dynamic INT8 (the ladder's export rungs). Keeps fp32 for rollback."""
import shutil

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

from shifa import config as C


def export(src, dst, auto_cls):
    model = auto_cls.from_pretrained(str(src)).eval()
    dst.mkdir(parents=True, exist_ok=True)
    ids = torch.ones(2, 16, dtype=torch.long)
    mask = torch.ones(2, 16, dtype=torch.long)
    torch.onnx.export(
        model, (ids, mask), str(dst / "model.onnx"), input_names=["input_ids", "attention_mask"],
        output_names=["logits"], opset_version=17, dynamo=False,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"} if auto_cls is AutoModelForSequenceClassification else {0: "batch", 1: "seq"}})
    for f in src.iterdir():           # tokenizer files, labels.json, card.json
        if f.name not in ("model.safetensors", "pytorch_model.bin", "training_args.bin", "optimizer.pt") and f.is_file():
            shutil.copy(f, dst / f.name)


def main():
    export(C.TOPIC_FP32, C.TOPIC_ONNX_FP32, AutoModelForSequenceClassification)
    # INT8: weights quantised dynamically; activations stay fp32 until the matmul. fp32 dir is the rollback.
    C.TOPIC_ONNX_INT8.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(C.TOPIC_ONNX_FP32 / "model.onnx"), str(C.TOPIC_ONNX_INT8 / "model.onnx"), weight_type=QuantType.QInt8)
    for f in C.TOPIC_ONNX_FP32.iterdir():
        if not f.name.startswith("model.onnx"):          # never copy fp32 weights/data into the INT8 folder
            shutil.copy(f, C.TOPIC_ONNX_INT8 / f.name)
    # NER stays fp32: boundary decisions are fragile under quantisation (decision documented in DECISIONS.md)
    if (C.NER_FP32 / "config.json").exists():
        export(C.NER_FP32, C.NER_ONNX_FP32, AutoModelForTokenClassification)
    for d in (C.TOPIC_ONNX_FP32, C.TOPIC_ONNX_INT8):
        print(d.name, round((d / "model.onnx").stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
