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
from PIL import Image
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_correct_inputs(processor, model_path):
    """Create inputs using the exact method that works from the minimal test"""
    print("=== CREATING CORRECT INPUTS ===")
    
    # Create a dummy image (same as minimal test)
    image = Image.new('RGB', (448, 448), color='blue')
    
    # Use the exact same format as the minimal test
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": "What is in this image?",
                },
            ],
        }
    ]
    
    # Apply chat template (same as minimal test)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Formatted prompt: {text}")
    
    # Prepare inputs (same as minimal test)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    
    print("Input shapes:")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    return inputs

def verify_and_fix_inputs(model, inputs):
    """Verify inputs and fix any issues"""
    print("=== VERIFYING AND FIXING INPUTS ===")
    
    # Ensure all required keys are present and properly formatted
    required_keys = ['input_ids', 'attention_mask', 'pixel_values']
    
    for key in required_keys:
        if key not in inputs:
            print(f"❌ Missing required key: {key}")
            return None
    
    # Check for image_grid_thw and ensure it's included
    if 'image_grid_thw' in inputs:
        print("✓ image_grid_thw found in inputs")
    else:
        print("⚠️ image_grid_thw not found, but will proceed")
    
    # Ensure tensors are the right type and device
    fixed_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            fixed_inputs[key] = value
        else:
            print(f"⚠️ Converting {key} to tensor")
            fixed_inputs[key] = torch.tensor(value)
    
    return fixed_inputs

def test_forward_pass(model, inputs):
    """Test forward pass with proper error handling"""
    print("=== TESTING FORWARD PASS ===")
    
    try:
        # Move inputs to model device
        inputs_device = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs_device[k] = v.to(model.device)
        
        # Add labels for training
        inputs_device["labels"] = inputs_device["input_ids"].clone()
        
        print("Inputs on device:")
        for key, value in inputs_device.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} on {value.device}")
        
        # Perform forward pass
        with torch.no_grad():
            outputs = model(**inputs_device)
        
        print(f"✓ Forward pass successful! Loss: {outputs.loss}")
        return True, outputs.loss
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--max_steps", type=int, default=1)
    args = parser.parse_args()
    
    print("=== FINAL QWEN3-VL TRAINING FIX ===")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    try:
        # Load processor and model (same order as minimal test)
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            local_files_only=True
        )
        print("✓ Processor loaded")
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("✓ Model loaded")
        
        # Create inputs using the method that worked in minimal test
        inputs = create_correct_inputs(processor, args.model_id)
        
        # Verify and fix inputs
        fixed_inputs = verify_and_fix_inputs(model, inputs)
        if fixed_inputs is None:
            print("❌ Input verification failed")
            return
        
        # Test forward pass
        forward_success, loss = test_forward_pass(model, fixed_inputs)
        if not forward_success:
            print("❌ Forward pass test failed - cannot proceed with training")
            return
        
        # Convert to training format
        processed = {}
        for key, value in fixed_inputs.items():
            if isinstance(value, torch.Tensor):
                # Remove batch dimension for dataset but keep the tensor structure
                if value.dim() > 1 and value.shape[0] == 1:
                    processed[key] = value.squeeze(0).cpu().numpy()
                else:
                    processed[key] = value.cpu().numpy()
        
        processed["labels"] = processed["input_ids"].copy()
        
        print("Processed dataset items:")
        for key, value in processed.items():
            print(f"  {key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
        
        # Create dataset
        train_dataset = Dataset.from_list([processed])
        print(f"✓ Created training dataset with 1 example")
        
        # Minimal training setup
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            remove_unused_columns=False,
            logging_steps=1,
            report_to=[],
            save_strategy="no",
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            dataloader_num_workers=0,
            seed=42,
            max_grad_norm=0.1,
            optim="adamw_torch",
            logging_first_step=True,
            gradient_accumulation_steps=1,
            prediction_loss_only=True,
            dataloader_pin_memory=False,
            eval_steps=None,
            save_steps=None,
            ddp_find_unused_parameters=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
        )
        
        print("\n=== STARTING TRAINING ===")
        
        # Clear memory before training
        torch.cuda.empty_cache()
        
        # Train for 1 step
        train_result = trainer.train()
        print("✓ Training completed successfully!")
        print(f"Training loss: {train_result.metrics['train_loss']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: Simple manual training
        print("\n=== ATTEMPTING MANUAL TRAINING FALLBACK ===")
        try:
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            
            # Use the fixed inputs
            inputs_device = {}
            for k, v in fixed_inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs_device[k] = v.to(model.device)
            
            inputs_device["labels"] = inputs_device["input_ids"].clone()
            
            # Forward pass
            outputs = model(**inputs_device)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"✓ Manual training step successful! Loss: {loss.item()}")
            
            # Save the model manually
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"✓ Model saved to {args.output_dir}")
            
        except Exception as e2:
            print(f"❌ Manual training also failed: {e2}")

if __name__ == "__main__":
    main()
