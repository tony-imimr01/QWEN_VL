import os
import torch
import argparse
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import Dataset
import json
from PIL import Image
import base64
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processor_safely(model_path):
    """Safely load processor with multiple fallback strategies"""
    print(f"Attempting to load processor from: {model_path}")
    
    # Strategy 1: Try direct local loading first
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True
        )
        print("✓ Processor loaded successfully from local directory")
        return processor
    except Exception as e:
        print(f"Direct local loading failed: {e}")
    
    # Strategy 2: Fallback to online download
    try:
        online_id = "Qwen/Qwen3-VL-4B-Instruct"
        print(f"Falling back to online download: {online_id}")
        processor = AutoProcessor.from_pretrained(
            online_id,
            trust_remote_code=True
        )
        processor.save_pretrained(model_path)
        print("✓ Processor downloaded and saved locally")
        return processor
    except Exception as e:
        print(f"Online download failed: {e}")
        raise

def decode_base64_image(image_str):
    """Decode base64 image string to PIL Image"""
    try:
        if image_str.startswith('data:image'):
            image_str = image_str.split(',', 1)[1]
        
        image_data = base64.b64decode(image_str)
        image = Image.open(BytesIO(image_data))
        return image.convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to decode image: {e}")
        return Image.new('RGB', (224, 224), color='red')

def preprocess_single_example(example, processor):
    """Process single example to avoid batch dimension issues"""
    try:
        conversations = example.get('conversations', [])
        if not conversations:
            return None
            
        # Take the first human message
        first_turn = conversations[0]
        if not isinstance(first_turn, dict) or 'value' not in first_turn:
            return None
            
        text = str(first_turn['value'])
        
        # Handle image
        image = None
        if 'image' in example and example['image']:
            image_data = example['image']
            if isinstance(image_data, str):
                image = decode_base64_image(image_data)
        
        # If no image but text has image token, use dummy image
        if image is None and '<image>' in text:
            image = Image.new('RGB', (224, 224), color='blue')
        elif image is None:
            # No image and no image token - skip or create simple text example
            image = Image.new('RGB', (224, 224), color='green')
            if '<image>' not in text:
                text = f"<image>\n{text}"
        
        # Ensure image token is present
        if image is not None and '<image>' not in text:
            text = f"<image>\n{text}"
        
        # Process single example
        inputs = processor(
            text=text,
            images=image,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Convert to format that datasets can handle
        processed = {}
        for key, value in inputs.items():
            # Remove batch dimension for individual examples
            if value.dim() > 1:
                processed[key] = value.squeeze(0).numpy()
            else:
                processed[key] = value.numpy()
        
        # Add labels
        processed["labels"] = processed["input_ids"].copy()
        
        return processed
        
    except Exception as e:
        logger.error(f"Error processing example: {e}")
        # Return a dummy example
        dummy_image = Image.new('RGB', (224, 224), color='red')
        inputs = processor(
            text=["<image>\nDescribe this image."],
            images=[dummy_image],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        processed = {}
        for key, value in inputs.items():
            if value.dim() > 1:
                processed[key] = value.squeeze(0).numpy()
            else:
                processed[key] = value.numpy()
        processed["labels"] = processed["input_ids"].copy()
        return processed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True) 
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--max_steps", type=int, default=2)
    args = parser.parse_args()
    
    print("Loading model and processor...")
    
    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load processor
    processor = load_processor_safely(args.model_id)
    
    print("Loading data...")
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    # Use only 1-2 examples for testing
    test_data = data[:2] if len(data) >= 2 else data
    
    print(f"Using {len(test_data)} examples for testing")
    
    # Create dataset
    dataset = Dataset.from_list(test_data)
    
    print("Preprocessing dataset (single example processing)...")
    
    # Process examples one by one to avoid batch dimension issues
    processed_data = []
    for i, example in enumerate(test_data):
        print(f"Processing example {i+1}/{len(test_data)}")
        processed = preprocess_single_example(example, processor)
        if processed is not None:
            processed_data.append(processed)
    
    if not processed_data:
        print("No valid examples processed. Creating dummy data...")
        dummy_image = Image.new('RGB', (224, 224), color='blue')
        inputs = processor(
            text=["<image>\nDescribe this image."],
            images=[dummy_image],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        processed = {}
        for key, value in inputs.items():
            if value.dim() > 1:
                processed[key] = value.squeeze(0).numpy()
            else:
                processed[key] = value.numpy()
        processed["labels"] = processed["input_ids"].copy()
        processed_data = [processed]
    
    # Create dataset from processed data
    processed_dataset = Dataset.from_list(processed_data)
    
    print(f"Processed dataset with {len(processed_dataset)} examples")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        remove_unused_columns=False,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        bf16=True,
        dataloader_num_workers=0,
        seed=42,
        max_grad_norm=0.01,
        optim="adamw_torch",
        ddp_timeout=1800,
        logging_first_step=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=default_data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
