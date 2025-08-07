# LLM stuff
import outlines
import torch
from transformers import AutoProcessor
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import json

# Image stuff
from PIL import Image
import requests

# Rich for pretty printing
from rich import print

# To use Qwen-2-VL:
from transformers import Qwen2VLForConditionalGeneration
model_name = "Qwen/Qwen2-VL-7B-Instruct"
model_class = Qwen2VLForConditionalGeneration

model = outlines.models.transformers_vision(
    model_name,
    model_class=model_class,
    model_kwargs={
        "device_map": "auto",
        "torch_dtype": torch.float32,
    },
    processor_kwargs={
        "device": "cpu", # set to "cpu" if you don't have a GPU
    },
)

def load_and_resize_image(image_path, max_size=1024):
    """
    Load and resize an image while maintaining aspect ratio

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) of the output image

    Returns:
        PIL Image: Resized image
    """
    image = Image.open(image_path)

    # Get current dimensions
    width, height = image.size

    # Calculate scaling factor
    scale = min(max_size / width, max_size / height)

    # Only resize if image is larger than max_size
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image
'''
# Path to the image
image_path = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/cookbook/images/trader-joes-receipt.jpg"

# Download the image
response = requests.get(image_path)
with open("receipt.png", "wb") as f:
    f.write(response.content)
'''
# Load + resize the image
image = load_and_resize_image("trader-joes-receipt.jpg")

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
    # Date is in the format YYYY-MM-DD. We can apply a regex pattern to ensure it's formatted correctly.
    date: Optional[str] = Field(pattern=r'\d{4}-\d{2}-\d{2}', description="Date in the format YYYY-MM-DD")
    payment_method: Literal["cash", "credit", "debit", "check", "other"]

# Set up the content you want to send to the model
messages = [
    {
        "role": "user",
        "content": [
            {
                # The image is provided as a PIL Image object
                "type": "image",
                "image": image,
            },
            {
                "type": "text",
                "text": f"""You are an expert at extracting information from receipts.
                Please extract the information from the receipt. Be as detailed as possible --
                missing or misreporting information is a crime.

                Return the information in the following JSON schema:
                {ReceiptSummary.model_json_schema()}
            """},
        ],
    }
]

# Convert the messages to the final prompt
processor = AutoProcessor.from_pretrained(model_name)
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Prepare a function to process receipts
receipt_summary_generator = outlines.generate.json(
    model,
    ReceiptSummary,

    # Greedy sampling is a good idea for numeric
    # data extraction -- no randomness.
    sampler=outlines.samplers.greedy()
)

# Generate the receipt summary
result = receipt_summary_generator(prompt, [image])
print(result)

# Save as JSON
with open("receipt_summary.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)