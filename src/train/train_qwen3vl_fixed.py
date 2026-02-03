import torch
import argparse
import os
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model
import numpy as np

def setup_lora(model):
    """Configure LoRA for efficient training"""
    lora_config = LoraConfig(
        r=8,  # Lower rank for memory efficiency
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}% of total)")
    return model

def create_training_data(processor):
    """Create training data with proper image_grid_thw"""
    examples = []
    
    # Create training examples
    training_data = [
        {
            "image_color": "blue",
            "question": "What color is this image?",
            "answer": "This image is completely blue with no other elements."
        },
        {
            "image_color": "red", 
            "question": "Describe what you see in this image.",
            "answer": "This is a solid red image, uniform in color with no objects or patterns."
        }
    ]
    
    for i, data in enumerate(training_data):
        # Create image
        image = Image.new('RGB', (224, 224), color=data["image_color"])
        
        # Create conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": data["question"]},
                ],
            },
            {
                "role": "assistant", 
                "content": data["answer"]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # Process inputs - this will include image_grid_thw
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
            max_length=256,
            truncation=True
        )
        
        # Ensure image_grid_thw is included
        if "image_grid_thw" not in inputs:
            print("WARNING: image_grid_thw not in inputs, creating default")
            # Create default image_grid_thw [temporal, height, width]
            inputs["image_grid_thw"] = torch.tensor([[1, 14, 14]])  # For 224x224 image with patch size 16
        
        # Convert to dataset format
        example = {
            "input_ids": inputs["input_ids"].squeeze(0).numpy(),
            "attention_mask": inputs["attention_mask"].squeeze(0).numpy(),
            "pixel_values": inputs["pixel_values"].squeeze(0).numpy(),
            "labels": inputs["input_ids"].squeeze(0).numpy().copy(),
        }
        
        # Add image_grid_thw if present
        if "image_grid_thw" in inputs:
            example["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0).numpy()
        
        examples.append(example)
        print(f"Created example {i+1}: {data['image_color']} image")
    
    return examples

def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL with LoRA")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=3)
    args = parser.parse_args()
    
    print("Starting Qwen3-VL Training...")
    
    # Clear memory
    torch.cuda.empty_cache()
    
    try:
        # Load processor and model
        print("Loading model and processor...")
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Model and processor loaded")
        
        # Apply LoRA
        print("Applying LoRA...")
        model = setup_lora(model)
        
        # Create training data
        print("Creating training data...")
        examples = create_training_data(processor)
        train_dataset = Dataset.from_list(examples)
        print(f"Training dataset created with {len(examples)} examples")
        
        # Check dataset columns
        print("Dataset columns:", train_dataset.column_names)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            logging_steps=1,
            save_strategy="no",
            bf16=True,
            remove_unused_columns=False,  # Important: keep all columns
            report_to=[],
            gradient_checkpointing=True,
            dataloader_drop_last=False,
            optim="adamw_torch",
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(processor.tokenizer, mlm=False),
        )
        
        # Train
        print("Starting training...")
        train_result = trainer.train()
        
        # Save
        trainer.save_model()
        processor.save_pretrained(args.output_dir)
        
        print("Training completed successfully!")
        print(f"Final loss: {train_result.metrics['train_loss']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
