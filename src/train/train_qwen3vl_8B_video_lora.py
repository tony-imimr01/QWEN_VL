#!/usr/bin/env python3
"""
Qwen3-VL-8B Video Training Script with LoRA Fine-tuning - CORRECT FORMAT
Handles the actual format of washhand.json with nested video paths in messages content
"""

import torch
import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path
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
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import random
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_training.log')
    ]
)
logger = logging.getLogger('Qwen3VL-Video-Trainer')

def print_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free, total = torch.cuda.mem_get_info()
        free = free / 1024**3
        total = total / 1024**3
        logger.info(f"💾 GPU Memory - Allocated: {alloc:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB, Total: {total:.2f}GB")
        return alloc, reserved, free, total
    return None

def extract_frame_from_video(video_path: str, timestamp_seconds: float = None, frame_number: int = None) -> Image.Image:
    """
    Extract a specific frame from video at given timestamp or frame number
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"🎥 Video info - FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        # Calculate frame number from timestamp
        if timestamp_seconds is not None:
            if timestamp_seconds < 0 or timestamp_seconds > duration:
                logger.warning(f"Timestamp {timestamp_seconds}s out of range [0, {duration:.2f}s]. Clamping to valid range.")
                timestamp_seconds = max(0, min(timestamp_seconds, duration))
            frame_number = int(timestamp_seconds * fps)
        elif frame_number is None:
            frame_number = 10  # Default to frame 10
        
        frame_number = max(0, min(frame_number, total_frames - 1))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_number} from video")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = image.resize((224, 224), Image.LANCZOS)
        
        logger.info(f"✅ Extracted frame {frame_number} at timestamp {frame_number/fps:.2f}s from {video_path}")
        return image
        
    except Exception as e:
        logger.error(f"❌ Error extracting frame: {e}")
        logger.error(traceback.format_exc())
        return Image.new('RGB', (224, 224), color='gray')

def load_training_data_correct_format(data_path: str) -> List[Dict[str, Any]]:
    """
    Load training data from JSON file with the actual format of washhand.json
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✅ Successfully loaded {len(data)} examples from {data_path}")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON decode error in {data_path}: {e}")
        logger.error(f"💡 Line {e.lineno}, column {e.colno}: {e.msg}")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load training data: {e}")
        logger.error(traceback.format_exc())
        raise

def extract_video_path_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extract video path from messages structure"""
    for message in messages:
        if message.get("role") == "user" and "content" in message:
            content = message["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video" and "video" in item:
                        return item["video"]
    return None

def extract_question_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extract question from messages structure"""
    for message in messages:
        if message.get("role") == "user" and "content" in message:
            content = message["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                        return item["text"]
            elif isinstance(content, str):
                return content
    return ""

def extract_answer_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extract answer from messages structure"""
    for message in messages:
        if message.get("role") == "assistant" and "content" in message:
            content = message["content"]
            if isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict) and "text" in content[0]:
                    return content[0]["text"]
                elif isinstance(content[0], str):
                    return content[0]
            elif isinstance(content, str):
                return content
    return ""

def create_training_examples_correct_format(processor, training_data: List[Dict[str, Any]], 
                                           video_dir: str, num_examples: int = -1) -> List[Dict[str, Any]]:
    """
    Create training examples from the actual washhand.json format with nested video paths
    """
    examples = []
    example_metadata = []
    
    if num_examples == -1 or num_examples > len(training_data):
        num_examples = len(training_data)
    
    logger.info(f"📊 Creating {num_examples} training examples from washhand.json...")
    
    # Process in batches to manage memory
    batch_size = 3
    for batch_start in range(0, num_examples, batch_size):
        batch_end = min(batch_start + batch_size, num_examples)
        logger.info(f"📦 Processing batch {batch_start+1} to {batch_end} of {num_examples}")
        
        for i in range(batch_start, batch_end):
            try:
                data = training_data[i]
                # Extract video path, question, and answer from messages
                messages = data.get("messages", [])
                video_path = extract_video_path_from_messages(messages)
                question = extract_question_from_messages(messages)
                answer = extract_answer_from_messages(messages)
                
                if not video_path:
                    logger.warning(f"Example {i+1} has no video content. Skipping.")
                    continue
                
                if not question or not answer:
                    logger.warning(f"Example {i+1} has empty question or answer. Skipping.")
                    continue
                
                # Get full video path
                full_video_path = os.path.join(video_dir, video_path)
                
                # Check if video file exists
                if not os.path.exists(full_video_path):
                    logger.warning(f"Video file not found: {full_video_path}. Skipping example {i+1}.")
                    continue
                
                # Extract frame from video
                logger.info(f"🎬 Processing example {i+1}/{num_examples}: {video_path}")
                image = extract_frame_from_video(full_video_path)
                
                # Create conversation messages
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question},
                        ],
                    },
                    {
                        "role": "assistant", 
                        "content": answer
                    }
                ]
                
                # Apply chat template
                text = processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                # Process inputs
                inputs = processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )
                
                # Add image_grid_thw if missing
                if "image_grid_thw" not in inputs:
                    inputs["image_grid_thw"] = torch.tensor([[1, 16, 16]], dtype=torch.long)
                
                # Create example
                example = {
                    "input_ids": inputs["input_ids"].squeeze(0).numpy(),
                    "attention_mask": inputs["attention_mask"].squeeze(0).numpy(),
                    "pixel_values": inputs["pixel_values"].squeeze(0).numpy(),
                    "image_grid_thw": inputs["image_grid_thw"].squeeze(0).numpy(),
                    "labels": inputs["input_ids"].squeeze(0).numpy().copy(),
                }
                
                metadata = {
                    "id": data.get("id", f"example_{i+1}"),
                    "question": question,
                    "answer": answer,
                    "video": video_path
                }
                
                examples.append(example)
                example_metadata.append(metadata)
                
                logger.info(f"✅ Created example {i+1}/{num_examples}: ID={data.get('id', 'N/A')}")
                
            except Exception as e:
                logger.error(f"❌ Error processing example {i+1}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Clear memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)  # Allow memory to stabilize
        print_memory_usage()
    
    logger.info(f"✅ Successfully created {len(examples)}/{num_examples} training examples")
    return examples, example_metadata

def setup_lora(model):
    """Configure LoRA for efficient training - optimized for 8B model"""
    try:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
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
        model.print_trainable_parameters()
        model.train()
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to setup LoRA: {e}")
        logger.error(traceback.format_exc())
        raise

class CustomDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator that handles image_grid_thw"""
    
    def __call__(self, features):
        batch = super().__call__(features)
        
        if "image_grid_thw" in features[0] and features[0]["image_grid_thw"] is not None:
            try:
                batch["image_grid_thw"] = torch.stack([
                    torch.tensor(f["image_grid_thw"]) for f in features
                ]).to(batch["input_ids"].device)
            except Exception as e:
                logger.warning(f"⚠️ Failed to add image_grid_thw to batch: {e}")
        
        return batch

def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL-8B on washhand.json Data with LoRA - CORRECT FORMAT")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to washhand.json file")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing training videos")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=8000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--num_examples", type=int, default=12, help="Number of training examples to use (12 for all examples)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("🎯 STARTING QWEN3-VL-8B TRAINING ON WASHHAND.JSON - CORRECT FORMAT")
    print("=" * 60)
    logger.info(f"📝 Arguments: {vars(args)}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 Cleared CUDA cache")
    print_memory_usage()
    
    try:
        start_time = time.time()
        
        # Step 1: Load training data with correct format
        logger.info("🔍 Loading washhand.json file...")
        training_data = load_training_data_correct_format(args.data_path)
        
        # Step 2: Load processor and model
        logger.info("📥 Loading 8B model and processor...")
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True
        )
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info("✅ 8B Model and processor loaded successfully")
        print_memory_usage()
        
        # Step 3: Apply LoRA
        logger.info("⚙️ Applying LoRA configuration for 8B model...")
        model = setup_lora(model)
        print_memory_usage()
        
        # Step 4: Create training examples from correct format
        logger.info("🎬 Creating training examples from washhand.json...")
        examples, example_metadata = create_training_examples_correct_format(
            processor=processor,
            training_data=training_data,
            video_dir=args.video_dir,
            num_examples=args.num_examples
        )
        
        if not examples:
            raise ValueError("No valid training examples were created")
        
        # Create dataset
        train_dataset = Dataset.from_list(examples)
        logger.info(f"✅ Training dataset created with {len(examples)} examples")
        
        # Save example info
        with open(os.path.join(args.output_dir, "training_examples.json"), "w", encoding='utf-8') as f:
            json.dump(example_metadata, f, indent=2, ensure_ascii=False)
        
        # Step 5: Setup training arguments
        logger.info("⚙️ Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            logging_steps=1,
            save_steps=max(1, args.max_steps // 5),
            save_total_limit=3,
            bf16=True if torch.cuda.is_available() else False,
            fp16=False if torch.cuda.is_available() else True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=["tensorboard"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_drop_last=False,
            optim="adamw_torch",
            max_grad_norm=0.3,
            warmup_steps=0,
            logging_dir=os.path.join(args.output_dir, "logs"),
            run_name=f"qwen3vl_8b_correct_{time.strftime('%Y%m%d_%H%M%S')}",
            disable_tqdm=False,
            log_level="info",
            prediction_loss_only=True,
        )
        
        # Step 6: Create trainer
        logger.info("🚀 Creating trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=CustomDataCollator(processor.tokenizer, mlm=False),
        )
        
        # Step 7: Train
        print("=" * 60)
        print("🎬 STARTING TRAINING FOR 8B MODEL...")
        print("=" * 60)
        logger.info(f"🎯 Training {len(examples)} examples for {args.max_steps} steps")
        logger.info(f"⚡ Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation_steps}")
        print_memory_usage()
        
        model.train()
        
        train_start_time = time.time()
        train_result = trainer.train()
        train_end_time = time.time()
        
        # Step 8: Save results
        logger.info("💾 Saving trained model and results...")
        trainer.save_model()
        processor.save_pretrained(args.output_dir)
        
        # Save training metrics
        training_time = train_end_time - train_start_time
        total_time = train_end_time - start_time
        
        metrics = {
            "train_loss": train_result.metrics.get('train_loss', 0.0),
            "training_time_seconds": training_time,
            "total_time_seconds": total_time,
            "samples_per_second": train_result.metrics.get('train_samples_per_second', 0.0),
            "model_size": "8B",
            "lora_rank": 8,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "num_examples": len(examples),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metrics_path = os.path.join(args.output_dir, "training_metrics.json")
        with open(metrics_path, "w", encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print("=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        logger.info(f"📊 Final training loss: {metrics['train_loss']:.4f}")
        logger.info(f"⏱️ Training time: {training_time:.2f} seconds")
        logger.info(f"💾 Model and results saved to: {args.output_dir}")
        print_memory_usage()
        
        return 0
        
    except Exception as e:
        print("=" * 60)
        print("❌ TRAINING FAILED!")
        print("=" * 60)
        logger.error(f"🔥 Critical error: {e}")
        logger.error(traceback.format_exc())
        
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "args": vars(args)
        }
        
        error_path = os.path.join(args.output_dir, "training_error.json")
        with open(error_path, "w", encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
        
        logger.error(f"❌ Error details saved to: {error_path}")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
