#!/usr/bin/env python3
"""
Qwen3-VL-7B Video Training Script with LoRA Fine-tuning
Complete solution for training on video frames with timestamp-based frame extraction
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
        free = torch.cuda.mem_get_info()[0] / 1024**3
        total = torch.cuda.mem_get_info()[1] / 1024**3
        logger.info(f"💾 GPU Memory - Allocated: {alloc:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB, Total: {total:.2f}GB")
        return alloc, reserved, free, total
    return None

def extract_frame_from_video(video_path: str, timestamp_seconds: float = None, frame_number: int = None) -> Image.Image:
    """
    Extract a specific frame from video at given timestamp or frame number
    
    Args:
        video_path: Path to video file
        timestamp_seconds: Timestamp in seconds (preferred over frame_number)
        frame_number: Frame number if timestamp not provided
        
    Returns:
        PIL Image of the extracted frame, resized to 224x224
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
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
            frame_number = 0  # Default to first frame
        
        # Ensure frame number is within valid range
        frame_number = max(0, min(frame_number, total_frames - 1))
        
        # Set position and read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_number} from video")
        
        # Convert BGR to RGB and create PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Resize to model input size
        image = image.resize((224, 224), Image.LANCZOS)
        
        logger.info(f"✅ Extracted frame {frame_number} at timestamp {frame_number/fps:.2f}s from {video_path}")
        return image
        
    except Exception as e:
        logger.error(f"❌ Error extracting frame: {e}")
        logger.warning("Using fallback synthetic image")
        # Fallback to synthetic image
        return Image.new('RGB', (224, 224), color='gray')

def load_training_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load training data from JSON file
    
    Args:
        data_path: Path to JSON file containing training data
        
    Returns:
        List of training examples with video timestamps
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate data structure
        for i, item in enumerate(data):
            if "video" not in item:
                raise ValueError(f"Example {i} missing 'video' field")
            if "conversations" not in item:
                raise ValueError(f"Example {i} missing 'conversations' field")
            if len(item["conversations"]) < 2:
                raise ValueError(f"Example {i} needs at least one human-assistant conversation pair")
        
        logger.info(f"✅ Successfully loaded {len(data)} training examples from {data_path}")
        return data
        
    except Exception as e:
        logger.error(f"❌ Failed to load training data: {e}")
        raise

def create_training_examples_from_video(processor, training_data: List[Dict[str, Any]], 
                                       video_dir: str, num_examples: int = -1) -> List[Dict[str, Any]]:
    """
    Create training examples from video data with frame extraction
    
    Args:
        processor: Model processor
        training_data: List of training examples from JSON
        video_dir: Base directory containing video files
        num_examples: Number of examples to create (-1 for all)
        
    Returns:
        List of processed training examples
    """
    examples = []
    
    if num_examples == -1 or num_examples > len(training_data):
        num_examples = len(training_data)
    
    logger.info(f"📊 Creating {num_examples} training examples from video data...")
    
    for i, data in enumerate(training_data[:num_examples]):
        try:
            # Get video path and frame info
            video_filename = data["video"]
            video_path = os.path.join(video_dir, video_filename)
            
            # Get timestamp - try multiple possible field names
            timestamp = None
            for key in ["timestamp_seconds", "timestamp", "time_seconds", "frame_time"]:
                if key in data:
                    timestamp = data[key]
                    break
            
            # Get frame number if timestamp not available
            frame_number = None
            if timestamp is None:
                for key in ["frame_number", "frame_idx"]:
                    if key in data:
                        frame_number = data[key]
                        break
            
            if timestamp is None and frame_number is None:
                logger.warning(f"Example {i+1} has no timestamp or frame number. Using first frame.")
            
            # Extract frame from video
            logger.info(f"🎬 Processing example {i+1}/{num_examples}: {video_filename}")
            image = extract_frame_from_video(video_path, timestamp_seconds=timestamp, frame_number=frame_number)
            
            # Extract conversation
            conversations = data["conversations"]
            
            # Find first human message
            human_idx = next((idx for idx, msg in enumerate(conversations) if msg.get("from") == "human"), None)
            if human_idx is None:
                raise ValueError(f"Example {i+1} has no human message")
            
            # Find corresponding assistant message
            gpt_idx = next((idx for idx, msg in enumerate(conversations) if msg.get("from") == "gpt"), None)
            if gpt_idx is None:
                raise ValueError(f"Example {i+1} has no gpt message")
            
            human_msg = conversations[human_idx]
            gpt_msg = conversations[gpt_idx]
            
            # Extract question and answer
            question = human_msg["value"].replace("<image>", "").strip()
            answer = gpt_msg["value"].strip()
            
            if not question or not answer:
                raise ValueError(f"Example {i+1} has empty question or answer")
            
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
            
            # Add image_grid_thw to prevent NoneType error
            if "image_grid_thw" not in inputs:
                # Default grid size for 224x224 image
                inputs["image_grid_thw"] = torch.tensor([[1, 14, 14]], dtype=torch.long)
            
            # Create example
            example = {
                "input_ids": inputs["input_ids"].squeeze(0).numpy(),
                "attention_mask": inputs["attention_mask"].squeeze(0).numpy(),
                "pixel_values": inputs["pixel_values"].squeeze(0).numpy(),
                "image_grid_thw": inputs["image_grid_thw"].squeeze(0).numpy(),
                "labels": inputs["input_ids"].squeeze(0).numpy().copy(),
                "id": data.get("id", f"example_{i+1}"),
                "question": question,
                "answer": answer,
                "video": video_filename,
                "timestamp": timestamp,
                "frame_number": frame_number
            }
            
            examples.append(example)
            logger.info(f"✅ Created example {i+1}/{num_examples}: ID={data.get('id', 'N/A')}")
            logger.debug(f"   Question: '{question[:50]}...'")
            logger.debug(f"   Answer: '{answer[:50]}...'")
            
            # Clear memory periodically
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
                print_memory_usage()
            
        except Exception as e:
            logger.error(f"❌ Error processing example {i+1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"✅ Successfully created {len(examples)}/{num_examples} training examples")
    return examples

def setup_lora(model):
    """Configure LoRA for efficient training - optimized for 7B model"""
    try:
        lora_config = LoraConfig(
            r=8,  # Rank for LoRA matrices
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
        
        # Print trainable parameters
        trainable_params = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                logger.debug(f"🔧 Trainable parameter: {name}")
        
        logger.info(f"📊 Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,} ({trainable_params/total_params:.2%})")
        model.print_trainable_parameters()
        
        # Ensure model is in training mode
        model.train()
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to setup LoRA: {e}")
        raise

class CustomDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator that handles image_grid_thw and video-specific features"""
    
    def __call__(self, features):
        batch = super().__call__(features)
        
        # Add image_grid_thw to the batch if present in features
        if "image_grid_thw" in features[0] and features[0]["image_grid_thw"] is not None:
            try:
                batch["image_grid_thw"] = torch.stack([
                    torch.tensor(f["image_grid_thw"]) for f in features
                ]).to(batch["input_ids"].device)
            except Exception as e:
                logger.warning(f"⚠️ Failed to add image_grid_thw to batch: {e}")
        
        return batch

def validate_video_files(training_data: List[Dict[str, Any]], video_dir: str) -> bool:
    """
    Validate that all required video files exist and are accessible
    
    Args:
        training_data: List of training examples
        video_dir: Directory containing video files
        
    Returns:
        True if all videos are valid, False otherwise
    """
    valid = True
    video_files = set()
    
    for data in training_data:
        video_filename = data["video"]
        video_files.add(video_filename)
    
    for video_filename in video_files:
        video_path = os.path.join(video_dir, video_filename)
        if not os.path.exists(video_path):
            logger.error(f"❌ Video file not found: {video_path}")
            valid = False
            continue
        
        # Try to open video to check if it's valid
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"❌ Could not open video file: {video_path}")
                valid = False
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                logger.info(f"✅ Video validated: {video_filename} - FPS: {fps:.2f}, Frames: {total_frames}, Duration: {duration:.2f}s")
            cap.release()
        except Exception as e:
            logger.error(f"❌ Error validating video {video_path}: {e}")
            valid = False
    
    return valid

def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL-7B on Video Data with LoRA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data JSON file")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing training videos")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--num_examples", type=int, default=-1, help="Number of training examples to use (-1 for all)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--validate_only", action="store_true", help="Only validate data and exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("🎯 STARTING QWEN3-VL-7B VIDEO TRAINING")
    print("=" * 60)
    logger.info(f"📝 Arguments: {vars(args)}")
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 Cleared CUDA cache")
    print_memory_usage()
    
    try:
        start_time = time.time()
        
        # Step 1: Validate video files first
        logger.info("🔍 Validating video files...")
        training_data = load_training_data(args.data_path)
        
        if not validate_video_files(training_data, args.video_dir):
            logger.error("❌ Video validation failed. Please fix video files and try again.")
            return 1
        
        if args.validate_only:
            logger.info("✅ Validation completed successfully. Exiting due to --validate_only flag.")
            return 0
        
        # Step 2: Load processor and model
        logger.info("📥 Loading 7B model and processor...")
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True
        )
        
        logger.info("🧠 Loading model with optimizations...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        logger.info("✅ 7B Model and processor loaded successfully")
        print_memory_usage()
        
        # Step 3: Apply LoRA
        logger.info("⚙️ Applying LoRA configuration for 7B model...")
        model = setup_lora(model)
        print_memory_usage()
        
        # Step 4: Create training examples from video data
        logger.info("🎬 Creating training examples from video frames...")
        examples = create_training_examples_from_video(
            processor=processor,
            training_data=training_data,
            video_dir=args.video_dir,
            num_examples=args.num_examples
        )
        
        if not examples:
            raise ValueError("No valid training examples were created from video data")
        
        # Create dataset
        train_dataset = Dataset.from_list(examples)
        logger.info(f"✅ Training dataset created with {len(examples)} examples")
        
        # Save example info for reference
        example_info = [{
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "video": ex["video"],
            "timestamp": ex["timestamp"],
            "frame_number": ex["frame_number"]
        } for ex in examples]
        
        with open(os.path.join(args.output_dir, "training_examples.json"), "w", encoding='utf-8') as f:
            json.dump(example_info, f, indent=2, ensure_ascii=False)
        
        # Step 5: Setup training arguments - optimized for 7B
        logger.info("⚙️ Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            logging_steps=1,
            save_steps=max(1, args.max_steps // 5),  # Save 5 times during training
            save_total_limit=3,
            bf16=True if torch.cuda.is_available() else False,
            fp16=False if torch.cuda.is_available() else True,
            dataloader_num_workers=0,  # Set to 0 for video data to avoid multiprocessing issues
            remove_unused_columns=False,
            report_to=["tensorboard"],
            # Memory optimizations for 7B
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_drop_last=False,
            optim="adamw_torch",
            max_grad_norm=0.3,
            warmup_steps=0,
            logging_dir=os.path.join(args.output_dir, "logs"),
            run_name=f"qwen3vl_7b_video_{time.strftime('%Y%m%d_%H%M%S')}",
            disable_tqdm=False,
            log_level="info",
            log_level_replica="info",
            prediction_loss_only=True,
        )
        
        # Step 6: Create trainer with custom data collator
        logger.info("🚀 Creating trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=CustomDataCollator(processor.tokenizer, mlm=False),
        )
        
        # Step 7: Train
        print("=" * 60)
        print("🎬 STARTING VIDEO TRAINING FOR 7B MODEL...")
        print("=" * 60)
        logger.info(f"🎯 Training {len(examples)} video examples for {args.max_steps} steps")
        logger.info(f"⚡ Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation_steps}")
        print_memory_usage()
        
        # Explicitly set model to training mode
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
            "model_size": "7B",
            "lora_rank": 8,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "num_examples": len(examples),
            "num_videos": len(set(ex["video"] for ex in examples)),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metrics_path = os.path.join(args.output_dir, "training_metrics.json")
        with open(metrics_path, "w", encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print("=" * 60)
        print("🎉 VIDEO TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        logger.info(f"📊 Final training loss: {metrics['train_loss']:.4f}")
        logger.info(f"⏱️ Training time: {training_time:.2f} seconds")
        logger.info(f"💾 Model and results saved to: {args.output_dir}")
        print_memory_usage()
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("🎉 TRAINING SUMMARY")
        logger.info("="*70)
        logger.info(f"✅ Status: SUCCESS")
        logger.info(f"📊 Loss: {metrics['train_loss']:.4f}")
        logger.info(f"⏱️ Total Time: {total_time:.2f} seconds")
        logger.info(f"💾 Output Directory: {args.output_dir}")
        logger.info(f"🎯 Examples Trained: {len(examples)}")
        logger.info(f"🎬 Unique Videos: {metrics['num_videos']}")
        logger.info(f"⚡ Throughput: {metrics['samples_per_second']:.2f} samples/second")
        logger.info(f"🔄 Gradient Accumulation: {args.gradient_accumulation_steps} steps")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        print("=" * 60)
        print("❌ VIDEO TRAINING FAILED!")
        print("=" * 60)
        logger.error(f"🔥 Critical error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Save error info
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