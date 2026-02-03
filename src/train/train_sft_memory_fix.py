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

# Clear GPU memory
torch.cuda.empty_cache()

def load_processor_safely(model_path):
    """Safely load processor with multiple fallback strategies"""
    print(f"Attempting to load processor from: {model_path}")
    
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
    """Process single example with proper image token formatting"""
    try:
        conversations = example.get('conversations', [])
        if not conversations:
            return None
            
        # Take the first human message
        first_turn = conversations[0]
        if not isinstance(first_turn, dict) or 'value' not in first_turn:
            return None
            
        original_text = str(first_turn['value'])
        
        # Handle image
        image = None
        if 'image' in example and example['image']:
            image_data = example['image']
            if isinstance(image_data, str):
                image = decode_base64_image(image_data)
        
        # If we have an image, ensure proper formatting
        if image is not None:
            # Use the exact format Qwen3-VL expects for image tokens
            if '<image>' not in original_text and '<video>' not in original_text:
                text = f"<image>\n{original_text}"
            else:
                text = original_text.replace('<video>', '<image>')
        else:
            # No image available, use dummy image but don't include image token
            image = Image.new('RGB', (224, 224), color='blue')
            text = original_text.replace('<image>', '').replace('<video>', '')
        
        print(f"Processed text: {text[:100]}...")
        print(f"Image present: {image is not None}")
        
        # Process with the processor
        inputs = processor(
            text=text,
            images=image,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,  # Reduced sequence length to save memory
        )
        
        # Convert to format that datasets can handle
        processed = {}
        for key, value in inputs.items():
            if value.dim() > 1:
                processed[key] = value.squeeze(0).numpy()
            else:
                processed[key] = value.numpy()
        
        # Add labels
        processed["labels"] = processed["input_ids"].copy()
        
        return processed
        
    except Exception as e:
        logger.error(f"Error processing example: {e}")
        # Return a dummy example with proper image token
        dummy_image = Image.new('RGB', (224, 224), color='red')
        dummy_text = "<image>\nDescribe this image."
        
        inputs = processor(
            text=dummy_text,
            images=dummy_image,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,
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
    
    print("Loading model with memory optimization...")
    
    # Load model with memory optimizations
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Reduce CPU memory during loading
    )
    
    # Load processor
    processor = load_processor_safely(args.model_id)
    
    print("Loading data...")
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    # Use only 1 example to reduce memory usage
    test_data = data[:1]
    
    print(f"Using {len(test_data)} example for testing (reduced for memory)")
    
    # Process examples one by one
    processed_data = []
    for i, example in enumerate(test_data):
        print(f"Processing example {i+1}/{len(test_data)}")
        processed = preprocess_single_example(example, processor)
        if processed is not None:
            processed_data.append(processed)
    
    if not processed_data:
        print("No valid examples processed. Creating dummy data...")
        dummy_image = Image.new('RGB', (224, 224), color='blue')
        dummy_text = "<image>\nDescribe this image."
        
        inputs = processor(
            text=dummy_text,
            images=dummy_image,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,
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
    
    print(f"Processed dataset with {len(processed_dataset)} example")
    
    # Memory-optimized training arguments
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
        max_grad_norm=0.1,  # Reduced gradient norm
        optim="adamw_torch_fused",  # Use fused optimizer for memory efficiency
        ddp_timeout=1800,
        logging_first_step=True,
        gradient_accumulation_steps=1,
        eval_steps=None,
        eval_delay=None,
        save_steps=None,
        save_total_limit=None,
        # Disable unnecessary features to save memory
        prediction_loss_only=True,
        # Reduce memory usage
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=default_data_collator,
    )
    
    print("Starting training with memory optimizations...")
    
    # Clear memory before training
    torch.cuda.empty_cache()
    
    # Train with gradient clipping disabled for this small test
    trainer.train()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
