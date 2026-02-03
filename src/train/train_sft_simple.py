import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
from datasets import Dataset
from PIL import Image
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--max_steps", type=int, default=1)
    args = parser.parse_args()
    
    print("=== SIMPLE QWEN3-VL TRAINING ===")
    
    # Clear memory
    torch.cuda.empty_cache()
    
    try:
        # Load model and processor
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ Model and processor loaded")
        
        # Create image and messages (using the exact format that worked)
        image = Image.new('RGB', (448, 448), color='blue')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Formatted prompt length: {len(text)}")
        
        # Process inputs
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        
        # Convert to dataset format
        example = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                example[key] = value.squeeze(0).numpy()  # Remove batch dimension
        example["labels"] = example["input_ids"].copy()
        
        # Create dataset
        train_dataset = Dataset.from_list([example])
        print(f"✓ Dataset created with 1 example")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            remove_unused_columns=False,
            logging_steps=1,
            save_strategy="no",
            bf16=True,
            dataloader_num_workers=0,
            gradient_accumulation_steps=1,
            report_to=[],
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        print("Starting training...")
        trainer.train()
        print("✓ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
