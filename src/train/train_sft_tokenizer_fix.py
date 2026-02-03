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

def setup_tokenizer_and_processor(model_path):
    """Properly setup tokenizer and processor with image token support"""
    print("=== SETTING UP TOKENIZER AND PROCESSOR ===")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True
    )
    
    print("✓ Tokenizer loaded")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    
    # Check if <image> is a special token
    if '<image>' not in tokenizer.added_tokens_encoder:
        print("⚠️  <image> is not a special token. This may cause issues.")
        print("Tokenizing '<image>':", tokenizer.tokenize('<image>'))
        print("Token IDs for '<image>':", tokenizer.encode('<image>'))
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        tokenizer=tokenizer,
        trust_remote_code=True,
        local_files_only=True
    )
    print("✓ Processor loaded")
    
    return processor, tokenizer

def find_image_token_id(tokenizer):
    """Find the correct image token ID that the model expects"""
    # Try different possible image token formats
    test_tokens = [
        '<image>', '<|image|>', '<image_token>', 
        '[IMAGE]', '<img>', '<vision>'
    ]
    
    for token in test_tokens:
        if token in tokenizer.added_tokens_encoder:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"✅ Found image token: '{token}' -> ID: {token_id}")
            return token_id
    
    # If no image token found, try to find it in the tokenizer's vocabulary
    print("❌ No standard image token found. Checking vocabulary...")
    
    # Look for tokens that might be image-related
    vocab = tokenizer.get_vocab()
    image_related = [k for k in vocab.keys() if 'image' in k.lower() or 'img' in k.lower() or 'vision' in k.lower()]
    
    if image_related:
        print(f"Possible image tokens: {image_related}")
        for token in image_related:
            token_id = vocab[token]
            print(f"  '{token}' -> ID: {token_id}")
        
        # Use the first one
        return vocab[image_related[0]]
    
    print("❌ No image token found in vocabulary!")
    return None

def create_working_inputs(processor, tokenizer, image_token_id=None):
    """Create inputs that definitely work with the model"""
    print("\n=== CREATING WORKING INPUTS ===")
    
    dummy_image = Image.new('RGB', (224, 224), color='blue')
    
    # Strategy 1: Try the processor's default method
    try:
        print("Trying processor with simple text...")
        inputs = processor(
            text="Describe this image",
            images=dummy_image,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        )
        print("✓ Processor method worked")
        return inputs
    except Exception as e:
        print(f"Processor method failed: {e}")
    
    # Strategy 2: Try with explicit image token
    if image_token_id is not None:
        try:
            print(f"Trying with image token ID {image_token_id}...")
            # Manually create inputs with image token
            text = "Describe this image"
            text_inputs = tokenizer(
                text,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=64,
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
            print("✓ Manual combination worked")
            return inputs
        except Exception as e:
            print(f"Manual combination failed: {e}")
    
    # Strategy 3: Use chat template if available
    if hasattr(processor, 'apply_chat_template'):
        try:
            print("Trying chat template...")
            messages = [
                {"role": "user", "content": "Describe this image"}
            ]
            formatted = processor.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=False
            )
            inputs = processor(
                text=formatted,
                images=dummy_image,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=64,
            )
            print("✓ Chat template worked")
            return inputs
        except Exception as e:
            print(f"Chat template failed: {e}")
    
    # Strategy 4: Last resort - text only
    print("Trying text-only approach...")
    inputs = tokenizer(
        "Describe this image",
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )
    inputs['pixel_values'] = torch.randn(1, 3, 224, 224)  # Dummy image features
    print("✓ Text-only approach (with dummy image)")
    return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--max_steps", type=int, default=1)
    args = parser.parse_args()
    
    print("=== TOKENIZER-FIXED TRAINING ===")
    
    # Load model first
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("✓ Model loaded")
    
    # Setup tokenizer and processor
    processor, tokenizer = setup_tokenizer_and_processor(args.model_id)
    
    # Find image token
    image_token_id = find_image_token_id(tokenizer)
    
    # Create working inputs
    inputs = create_working_inputs(processor, tokenizer, image_token_id)
    
    # Convert to training format
    processed = {}
    for key, value in inputs.items():
        if value.dim() > 1:
            processed[key] = value.squeeze(0).numpy()
        else:
            processed[key] = value.numpy()
    
    processed["labels"] = processed["input_ids"].copy()
    
    # Create dataset
    processed_dataset = Dataset.from_list([processed])
    print(f"Created dataset with 1 example")
    
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
        trainer.train()
        print("✓ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        
        # Try a simple forward pass
        print("\n=== ATTEMPTING SIMPLE FORWARD PASS ===")
        try:
            with torch.no_grad():
                # Use the same inputs but ensure they're on the right device
                inputs_device = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs_device[k] = v.to(model.device)
                    else:
                        inputs_device[k] = torch.tensor(v).to(model.device)
                
                inputs_device["labels"] = inputs_device["input_ids"].clone()
                
                outputs = model(**inputs_device)
                print(f"✓ Forward pass successful! Loss: {outputs.loss}")
        except Exception as e2:
            print(f"❌ Forward pass also failed: {e2}")
            print("\nThis suggests a fundamental incompatibility between the model and processor.")
            print("Possible solutions:")
            print("1. Re-download the model and processor from scratch")
            print("2. Use a different model version")
            print("3. Check if there are missing files in the model directory")

if __name__ == "__main__":
    main()
