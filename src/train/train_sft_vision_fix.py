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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_proper_qwen3vl_inputs(processor, model_path):
    """Create inputs that properly work with Qwen3-VL model"""
    print("=== CREATING PROPER QWEN3-VL INPUTS ===")
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (448, 448), color='blue')
    
    # Method 1: Use the proper message format for Qwen3-VL
    try:
        print("Trying Qwen3-VL message format...")
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": dummy_image},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            }
        ]
        
        inputs = processor(
            messages,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        print("✓ Qwen3-VL message format worked")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")
        return inputs
        
    except Exception as e:
        print(f"Qwen3-VL message format failed: {e}")
    
    # Method 2: Try text + images approach
    try:
        print("Trying text + images approach...")
        inputs = processor(
            text="Describe this image in detail.",
            images=[dummy_image],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        print("✓ Text + images approach worked")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")
        return inputs
        
    except Exception as e:
        print(f"Text + images approach failed: {e}")
    
    # Method 3: Manual construction as last resort
    print("Trying manual construction...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Manually create the input with vision tokens
        # Qwen3-VL expects: <|im_start|>user<|im_end|><|vision_start|>...image features...<|vision_end|><|im_start|>assistant<|im_end|>
        text = "<|im_start|>user\nDescribe this image<|im_end|><|im_start|>assistant\n"
        text_inputs = tokenizer(
            text,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        # Get image inputs
        image_inputs = processor.image_processor(
            dummy_image,
            return_tensors="pt"
        )
        
        # Combine manually
        inputs = {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'pixel_values': image_inputs['pixel_values'],
        }
        print("✓ Manual construction worked")
        return inputs
        
    except Exception as e:
        print(f"Manual construction failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--max_steps", type=int, default=1)
    args = parser.parse_args()
    
    print("=== QWEN3-VL VISION-FIXED TRAINING ===")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Load processor first
    processor = AutoProcessor.from_pretrained(
        args.model_id,
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
    inputs = create_proper_qwen3vl_inputs(processor, args.model_id)
    
    # Verify we have both text and image inputs
    required_keys = ['input_ids', 'attention_mask', 'pixel_values']
    for key in required_keys:
        if key not in inputs:
            print(f"❌ Missing required key: {key}")
            return
    
    print(f"✓ All required inputs present:")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")
        else:
            print(f"  {key}: {type(value)}")
    
    # Convert to training format
    processed = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if value.dim() > 1:
                processed[key] = value.squeeze(0).numpy()
            else:
                processed[key] = value.numpy()
        else:
            processed[key] = value
    
    processed["labels"] = processed["input_ids"].copy()
    
    # Create dataset
    processed_dataset = Dataset.from_list([processed])
    print(f"✓ Created dataset with 1 example")
    
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
        bf16=True,
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
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=default_data_collator,
    )
    
    print("\n=== STARTING TRAINING ===")
    
    # Clear memory
    torch.cuda.empty_cache()
    
    try:
        # Test forward pass first
        print("Testing forward pass...")
        with torch.no_grad():
            inputs_device = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs_device[k] = v.to(model.device)
            
            inputs_device["labels"] = inputs_device["input_ids"].clone()
            
            outputs = model(**inputs_device)
            print(f"✓ Forward pass successful! Loss: {outputs.loss}")
        
        # Now train
        trainer.train()
        print("✓ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        
        # More detailed error analysis
        print("\n=== DETAILED ERROR ANALYSIS ===")
        print("1. Check if model supports the input format")
        print("2. Verify vision encoder is properly connected")
        print("3. Check model configuration:")
        print(f"   - Model type: {type(model)}")
        print(f"   - Model device: {model.device}")
        print(f"   - Model dtype: {model.dtype}")
        
        if hasattr(model, 'config'):
            config = model.config
            print(f"   - Model name: {getattr(config, '_name_or_path', 'N/A')}")
            print(f"   - Vision config: {getattr(config, 'vision_config', 'N/A')}")
            print(f"   - Text config: {getattr(config, 'text_config', 'N/A')}")

if __name__ == "__main__":
    main()
