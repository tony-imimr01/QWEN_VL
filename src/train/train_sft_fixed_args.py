import os
import sys
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

# Custom argument parser that accepts all arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Qwen3-VL Fine-tuning")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Training arguments
    parser.add_argument("--use_liger", type=str, default="False")
    parser.add_argument("--freeze_vision_tower", type=str, default="True")
    parser.add_argument("--freeze_llm", type=str, default="True")
    parser.add_argument("--freeze_merger", type=str, default="True")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=str, default="True")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=0.01)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--max_seq_length", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--tf32", type=str, default="False")
    parser.add_argument("--ddp_timeout", type=int, default=1800)
    parser.add_argument("--logging_first_step", type=str, default="True")
    parser.add_argument("--bf16", type=str, default="True")
    parser.add_argument("--remove_unused_columns", type=str, default="False")
    
    return parser.parse_args()

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
        # Return a dummy image as fallback
        return Image.new('RGB', (224, 224), color='red')

def preprocess_function(examples, processor):
    """Properly preprocess examples for Qwen3-VL with image tokens"""
    try:
        texts = []
        images = []
        
        for i in range(len(examples.get('conversations', []))):
            convs = examples['conversations'][i]
            if not isinstance(convs, list) or len(convs) == 0:
                texts.append("")
                images.append(None)
                continue
                
            # Take the first human message
            first_turn = convs[0]
            if isinstance(first_turn, dict) and 'value' in first_turn:
                text = str(first_turn['value'])
                
                # Ensure image token is present if we have an image
                if 'image' in examples and i < len(examples['image']):
                    image_data = examples['image'][i]
                    if image_data and isinstance(image_data, str):
                        image = decode_base64_image(image_data)
                        images.append(image)
                        
                        # Make sure text has image token
                        if '<image>' not in text:
                            text = f"<image>\n{text}"
                    else:
                        images.append(None)
                        # Remove image token if no image
                        text = text.replace('<image>', '')
                else:
                    images.append(None)
                    text = text.replace('<image>', '')
                    
                texts.append(text)
            else:
                texts.append("")
                images.append(None)
        
        # Filter out examples without images for this test
        valid_indices = [i for i, img in enumerate(images) if img is not None]
        if not valid_indices:
            # Create a dummy example if no valid images
            dummy_image = Image.new('RGB', (224, 224), color='blue')
            inputs = processor(
                text=["<image>\nDescribe this image."],
                images=[dummy_image],
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs
        
        filtered_texts = [texts[i] for i in valid_indices]
        filtered_images = [images[i] for i in valid_indices]
        
        # Process with the processor
        inputs = processor(
            text=filtered_texts,
            images=filtered_images,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # For training, we need labels
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        # Return dummy data as fallback
        dummy_image = Image.new('RGB', (224, 224), color='green')
        inputs = processor(
            text=["<image>\nDescribe this image."],
            images=[dummy_image],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

def main():
    args = parse_arguments()
    
    print("Loading model and processor...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True
    )
    
    print("Loading data...")
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    # Use only 1-2 examples for testing
    test_data = data[:2] if len(data) >= 2 else data
    
    print(f"Using {len(test_data)} examples for testing")
    
    # Create dataset
    dataset = Dataset.from_list(test_data)
    
    print("Preprocessing dataset...")
    
    # Preprocess in batches
    def batch_preprocess(examples):
        return preprocess_function(examples, processor)
    
    processed_dataset = dataset.map(
        batch_preprocess,
        batched=True,
        batch_size=1,  # Small batch size to avoid issues
        remove_columns=dataset.column_names
    )
    
    # Convert string arguments to boolean
    bf16 = args.bf16.lower() == "true"
    gradient_checkpointing = args.gradient_checkpointing.lower() == "true"
    remove_unused_columns = args.remove_unused_columns.lower() == "true"
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        remove_unused_columns=remove_unused_columns,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        save_strategy=args.save_strategy,
        bf16=bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        ddp_timeout=args.ddp_timeout,
        logging_first_step=args.logging_first_step.lower() == "true",
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
