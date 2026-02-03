import os
import torch
import argparse
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import Dataset
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def manual_qwen3vl_processing(processor, tokenizer, model_path):
    """Manually create inputs that work with Qwen3-VL by adding vision tokens"""
    print("=== MANUAL QWEN3-VL PROCESSING ===")
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (448, 448), color='red')
    
    # Get the vision token IDs from the tokenizer
    vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    
    print(f"Vision token IDs: vision_start={vision_start_id}, vision_end={vision_end_id}")
    print(f"IM token IDs: im_start={im_start_id}, im_end={im_end_id}")
    
    # Method 1: Manual construction with explicit vision tokens
    try:
        print("Method 1: Manual construction with vision tokens...")
        
        # Create text with explicit vision tokens
        text = f"<|im_start|>user\n<|vision_start|><|vision_end|>Describe this image in detail.<|im_end|><|im_start|>assistant\n"
        
        # Tokenize the text
        text_inputs = tokenizer(
            text,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        # Process the image
        image_inputs = processor.image_processor(
            dummy_image,
            return_tensors="pt"
        )
        
        # Combine inputs
        inputs = {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'pixel_values': image_inputs['pixel_values'],
        }
        
        # Add image grid info if available
        if hasattr(processor.image_processor, 'image_grid_thw'):
            inputs['image_grid_thw'] = processor.image_processor.image_grid_thw
            
        print(f"✓ Manual construction worked")
        print(f"Input IDs: {inputs['input_ids'].shape}")
        print(f"Pixel values: {inputs['pixel_values'].shape}")
        return inputs
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Use processor with explicit image token markers
    try:
        print("Method 2: Processor with image markers...")
        
        # Try using the processor with a text that contains the vision tokens
        text_with_tokens = f"<|vision_start|><|vision_end|>Describe this image in detail."
        
        inputs = processor(
            text=text_with_tokens,
            images=dummy_image,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        print(f"✓ Processor with explicit tokens worked")
        return inputs
        
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Method 3: Use the model's built-in chat template
    try:
        print("Method 3: Using model's chat template...")
        
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": dummy_image},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            }
        ]
        
        # Apply chat template to get the formatted text
        formatted_text = processor.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        
        print(f"Formatted text: {formatted_text}")
        
        # Now process with the formatted text
        inputs = processor(
            text=formatted_text,
            images=dummy_image,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        print(f"✓ Chat template method worked")
        return inputs
        
    except Exception as e:
        print(f"Method 3 failed: {e}")
    
    raise Exception("All methods failed to create proper inputs")

def verify_model_forward_pass(model, inputs):
    """Verify that the model can perform a forward pass with the inputs"""
    print("=== VERIFYING FORWARD PASS ===")
    
    try:
        # Move inputs to model device
        inputs_device = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs_device[k] = v.to(model.device)
        
        # Add labels for training
        inputs_device["labels"] = inputs_device["input_ids"].clone()
        
        # Perform forward pass
        with torch.no_grad():
            outputs = model(**inputs_device)
        
        print(f"✓ Forward pass successful! Loss: {outputs.loss}")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--max_steps", type=int, default=1)
    args = parser.parse_args()
    
    print("=== COMPREHENSIVE QWEN3-VL FIX ===")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Load tokenizer and processor separately
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=True
    )
    print("✓ Tokenizer loaded")
    
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        tokenizer=tokenizer,
        trust_remote_code=True,
        local_files_only=True
    )
    print("✓ Processor loaded")
    
    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("✓ Model loaded")
    
    # Create proper inputs
    inputs = manual_qwen3vl_processing(processor, tokenizer, args.model_id)
    
    # Verify inputs have required keys
    required_keys = ['input_ids', 'attention_mask', 'pixel_values']
    for key in required_keys:
        if key not in inputs:
            print(f"❌ Missing required key: {key}")
            return
    
    print(f"✓ Input verification passed:")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")
    
    # Verify forward pass
    if not verify_model_forward_pass(model, inputs):
        print("❌ Forward pass verification failed - cannot proceed with training")
        return
    
    # Convert to training format
    processed = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            # Remove batch dimension for dataset
            if value.dim() > 1 and value.shape[0] == 1:
                processed[key] = value.squeeze(0).numpy()
            else:
                processed[key] = value.numpy()
        else:
            processed[key] = value
    
    processed["labels"] = processed["input_ids"].copy()
    
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
    
    try:
        # Train for 1 step
        train_result = trainer.train()
        print("✓ Training completed successfully!")
        print(f"Training loss: {train_result.metrics['train_loss']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        
        # Try alternative training approach with custom forward
        print("\n=== ATTEMPTING ALTERNATIVE TRAINING APPROACH ===")
        try:
            # Simple manual training loop
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            
            # Prepare inputs
            inputs_device = {}
            for k, v in inputs.items():
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
            
        except Exception as e2:
            print(f"❌ Manual training also failed: {e2}")
            print("\nThis suggests a fundamental issue with the model configuration.")
            print("Possible solutions:")
            print("1. Check if the model files are complete and not corrupted")
            print("2. Try a different model version or checkpoint")
            print("3. Consult the model's documentation for proper usage")

if __name__ == "__main__":
    main()
