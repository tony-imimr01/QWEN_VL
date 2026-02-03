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
from peft import LoraConfig, get_peft_model, TaskType
import json

def print_memory_usage():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"💾 GPU Memory - Allocated: {alloc:.2f}GB, Reserved: {reserved:.2f}GB")

def setup_lora(model):
    """Configure LoRA for efficient training"""
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "qkv_proj", "gate_up_proj", "lm_head"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Ensure model is in training mode and parameters require grad
    model.train()
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad = True
    
    return model

def create_training_examples(processor, num_examples=2):
    """Create training examples with different images and questions"""
    examples = []
    
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
    
    for i, data in enumerate(training_data[:num_examples]):
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
        
        # Process inputs
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Add image_grid_thw to prevent NoneType error
        inputs["image_grid_thw"] = torch.tensor([[1, 16, 16]], dtype=torch.long)
        
        # FIX: Remove requires_grad assignment - inputs don't need gradients
        examples.append({
            "input_ids": inputs["input_ids"].squeeze(0).numpy(),
            "attention_mask": inputs["attention_mask"].squeeze(0).numpy(),
            "pixel_values": inputs["pixel_values"].squeeze(0).numpy(),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0).numpy(),
            "labels": inputs["input_ids"].squeeze(0).numpy().copy(),  # Copy for labels
        })
        
        print(f"✅ Created example {i+1}: {data['image_color']} image - '{data['question']}'")
    
    return examples

def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL with LoRA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=3, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    args = parser.parse_args()
    
    print("================================================")
    print("🎯 Starting Qwen3-VL LoRA Training")
    print("================================================")
    
    # Clear memory
    torch.cuda.empty_cache()
    print_memory_usage()
    
    try:
        # Step 1: Load processor and model
        print("📥 Loading model and processor...")
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Model and processor loaded")
        print_memory_usage()
        
        # Step 2: Apply LoRA
        print("⚙️ Applying LoRA configuration...")
        model = setup_lora(model)
        print_memory_usage()
        
        # Step 3: Create training data
        print("📊 Creating training data...")
        examples = create_training_examples(processor, num_examples=2)
        
        # Create dataset
        train_dataset = Dataset.from_list(examples)
        print(f"✅ Training dataset created with {len(examples)} examples")
        
        # Custom data collator to handle image_grid_thw
        class CustomDataCollator(DataCollatorForLanguageModeling):
            def __call__(self, features):
                batch = super().__call__(features)
                # Add image_grid_thw to the batch
                if "image_grid_thw" in features[0]:
                    batch["image_grid_thw"] = torch.stack([
                        torch.tensor(f["image_grid_thw"]) for f in features
                    ])
                return batch
        
        # Step 4: Setup training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            logging_steps=1,
            save_steps=args.max_steps,
            save_total_limit=1,
            bf16=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=[],
            # Memory optimizations
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_drop_last=False,
            optim="adamw_torch",
            max_grad_norm=0.3,
            warmup_steps=0,
            logging_dir=os.path.join(args.output_dir, "logs"),
        )
        
        # Step 5: Create trainer with custom data collator
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=CustomDataCollator(processor.tokenizer, mlm=False),
        )
        
        # Step 6: Train
        print("================================================")
        print("🎬 Starting Training...")
        print("================================================")
        print_memory_usage()
        
        # Explicitly set model to training mode
        model.train()
        
        train_result = trainer.train()
        
        # Step 7: Save results
        print("💾 Saving results...")
        trainer.save_model()
        processor.save_pretrained(args.output_dir)
        
        # Save training metrics
        metrics = {
            "train_loss": train_result.metrics['train_loss'],
            "training_steps": train_result.metrics['train_runtime'],
            "samples_per_second": train_result.metrics['train_samples_per_second'],
        }
        
        with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("================================================")
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("================================================")
        print(f"📊 Final loss: {train_result.metrics['train_loss']:.4f}")
        print(f"💾 Model saved to: {args.output_dir}")
        print_memory_usage()
        
        return 0
        
    except Exception as e:
        print("================================================")
        print("❌ TRAINING FAILED!")
        print("================================================")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
