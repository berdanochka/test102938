# LLM stuff
import os
import outlines
import torch
from transformers import AutoProcessor
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from PIL import Image
import json
import os

# ======= CPU perf knobs (safe, output-identical) =======
# Let PyTorch use all CPU cores efficiently.
# Tune these two if you see oversubscription (start with your physical core count).
NUM_THREADS = max(1, os.cpu_count() or 1)
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(max(1, NUM_THREADS // 2))

# If you have MKL/OpenBLAS, these help avoid oversubscription too.
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

# ======= Model setup =======
from transformers import Qwen2VLForConditionalGeneration
model_name = "Qwen/Qwen2-VL-7B-Instruct"
model_class = Qwen2VLForConditionalGeneration

# Keep fp32 on CPU to guarantee identical results.
model = outlines.models.transformers_vision(
    model_name,
    model_class=model_class,
    model_kwargs={
        "device_map": "auto",
        "torch_dtype": torch.float32,
    },
    processor_kwargs={
        "device": "cpu",  # stay CPU; no accuracy changes from device shenanigans
    },
)

# Put the model in inference mode (graph breaks avoided, small speed win).
if hasattr(model, "eval"):
    model.eval()

# Processor can be reused
processor = AutoProcessor.from_pretrained(model_name)

def load_and_resize_image(image_path, max_size=1024):
    """
    Load and resize an image while maintaining aspect ratio
    """
    image = Image.open(image_path)

    width, height = image.size
    scale = min(max_size / width, max_size / height)

    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image

# -------- Schema & types --------
class Item(BaseModel):
    name: str
    quantity: Optional[int]
    price_per_unit: Optional[float]
    total_price: Optional[float]

class ReceiptSummary(BaseModel):
    store_name: str
    store_address: str
    store_number: Optional[int]
    items: List[Item]
    tax: Optional[float]
    total: Optional[float]
    date: Optional[str] = Field(pattern=r'\d{4}-\d{2}-\d{2}', description="Date in the format YYYY-MM-DD")
    payment_method: Literal["cash", "credit", "debit", "check", "other"]

# Cache the schema string so we don’t recompute it every call
RECEIPT_SCHEMA_JSON = ReceiptSummary.model_json_schema()

# Prebuild the (deterministic) JSON generator & sampler once
greedy = outlines.samplers.greedy()
receipt_summary_generator = outlines.generate.json(
    model,
    ReceiptSummary,
    sampler=greedy
)

def extract_receipt_to_json(image_path: str) -> dict:
    image = load_and_resize_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "You are an expert at extracting information from receipts. "
                        "Please extract the information from the receipt. Be as detailed as possible -- "
                        "missing or misreporting information is a crime.\n\n"
                        "Return the information in the following JSON schema:\n"
                        f"{RECEIPT_SCHEMA_JSON}"
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # inference_mode gives you no-grad plus extra speed; output is identical.
    with torch.inference_mode():
        return receipt_summary_generator(prompt, [image])

if __name__ == "__main__":
    image_file = "trader-joes-receipt.jpg"
    result = extract_receipt_to_json(image_file)
    print(result)

    # Save result to JSON file
    json_filename = os.path.splitext(os.path.basename(image_file))[0] + ".json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved receipt data to {json_filename}")
