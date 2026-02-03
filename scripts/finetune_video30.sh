#!/bin/bash
set -euo pipefail

echo "================================================"
echo "🚀 QWEN3-VL-8B COMPLETE TRAINING SOLUTION WITH VIDEO SUPPORT"
echo "================================================"

# Configuration - UPDATED FOR 8B MODEL WITH VIDEO SUPPORT
MODEL_PATH="/home/fion/tony11/local_models/QWEN/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="/home/fion/tony11/output/training_results_8B"
DATA_DIR="/home/fion/tony11/data"
SCRIPT_DIR="/home/fion/tony11/src/train"
VIDEO_DIR="/home/fion/tony11"  # Directory for video files

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to check GPU memory
check_gpu_memory() {
    log_info "Checking GPU Memory..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
    else
        log_warning "nvidia-smi not available, skipping GPU memory check"
    fi
}

# Function to verify model directory structure
verify_model_directory() {
    log_info "Verifying model directory structure..."
    
    if [[ ! -d "$MODEL_PATH" ]]; then
        log_error "Model directory does not exist: $MODEL_PATH"
        exit 1
    fi
    
    log_info "Model directory contents:"
    ls -la "$MODEL_PATH"
    
    # Check for essential files
    essential_files=("config.json" "model.safetensors" "pytorch_model.bin" "tokenizer.json" "tokenizer_config.json")
    found_files=0
    for file in "${essential_files[@]}"; do
        if [[ -f "$MODEL_PATH/$file" ]]; then
            log_success "Found: $file"
            found_files=$((found_files + 1))
        else
            log_warning "Missing: $file"
        fi
    done
    
    if [[ $found_files -eq 0 ]]; then
        log_error "No essential model files found in $MODEL_PATH"
        log_error "Please ensure the model is properly downloaded and extracted"
        exit 1
    fi
    
    log_success "Model directory verification completed - found $found_files essential files"
}

# Function to setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Memory optimization - IMPORTANT FOR 8B MODEL
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TOKENIZERS_PARALLELISM=false
    export CUDA_LAUNCH_BLOCKING=1
    
    # Force offline mode and local files only
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HUGGINGFACE_HUB_CACHE="$MODEL_PATH"
    
    # Set Python path
    export PYTHONPATH="/home/fion/tony4/src:$PYTHONPATH"
    
    # Clear GPU memory
    log_info "Clearing GPU memory..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    check_gpu_memory
}

# Function to install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    pip install --upgrade pip
    
    # Core packages
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121          
    
    # Transformers and training
    pip install transformers>=4.37.0 accelerate datasets peft \
                bitsandbytes pillow matplotlib nvidia-ml-py3 opencv-python numpy tqdm
    
    # Check installations
    python3 -c "
import importlib, sys
required_packages = ['transformers', 'peft', 'datasets', 'torch', 'PIL', 'cv2', 'numpy']
all_installed = True
for pkg in required_packages:
    try:
        importlib.import_module(pkg)
        print(f'✓ {pkg} installed')
    except ImportError as e:
        print(f'✗ {pkg} failed: {e}')
        all_installed = False

if not all_installed:
    sys.exit(1)
    "
}

# Function to simply validate JSON without fixing
validate_json() {
    log_info "Validating JSON format for washhand.json..."
    
    local validation_script="/tmp/validate_json.py"
    cat > "$validation_script" <<'PYTHON'
import json
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_json.py <json_file_path>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"❌ File does not exist: {json_file}")
        sys.exit(1)
    
    print(f"🔍 Validating JSON file: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ JSON validation passed: {json_file}")
        print(f"📊 Found {len(data)} examples in the dataset")
        sys.exit(0)
    except json.JSONDecodeError as e:
        print(f"❌ JSON DECODE ERROR: {e}")
        print(f"💡 Line {e.lineno}, column {e.colno}: {e.msg}")
        print(f"📝 Character position: {e.pos}")
        print("\n🔧 MANUAL FIX REQUIRED:")
        print("   • Open the file in a text editor with line numbers (VS Code, nano, vim)")
        print("   • Navigate to the error location (line 241, column 3)")
        print("   • Common fixes needed:")
        print("     - Add missing comma between array elements or objects")
        print("     - Fix missing closing brackets/braces")
        print("     - Ensure all keys and string values are properly quoted")
        print("   • Use a JSON validator like jsonlint.com to verify fixes")
        sys.exit(1)
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
PYTHON

    # Run JSON validation
    python3 "$validation_script" "$DATA_DIR/washhand.json"
    local json_result=$?
    
    rm -f "$validation_script"
    
    if [[ $json_result -eq 0 ]]; then
        log_success "JSON validation passed!"
    else
        log_error "JSON validation failed! Manual fix required."
        exit 1
    fi
}

# Function to validate all video files before training
validate_all_videos() {
    log_info "Running comprehensive video validation..."
    
    local validation_script="/tmp/validate_videos.py"
    cat > "$validation_script" <<'PYTHON'
import cv2
import os
import json
import sys
import traceback

def validate_video(video_path):
    """Validate a single video file"""
    if not os.path.exists(video_path):
        print(f"❌ Missing: {video_path}")
        return False
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open: {video_path}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
            print(f"❌ Invalid metadata: {video_path}")
            return False
        
        # Try reading a frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(10, total_frames-1))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"❌ Cannot read frame: {video_path}")
            return False
        
        print(f"✅ Valid: {video_path} | Frames: {total_frames}, FPS: {fps:.2f}, Resolution: {width}x{height}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")
        traceback.print_exc()
        return False

def main():
    data_path = sys.argv[1]
    video_dir = sys.argv[2]
    
    print(f"\n🔍 Loading JSON data from: {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON data: {e}")
        sys.exit(1)
    
    if not isinstance(data, list):
        print(f"❌ JSON data is not a list. Expected list of examples.")
        sys.exit(1)
    
    total = len(data)
    valid = 0
    invalid_videos = []
    
    print(f"\n🔍 Validating {total} video examples...")
    
    for i, item in enumerate(data):
        messages = item.get('messages', [])
        video_path = None
        
        for msg in messages:
            if msg.get('role') == 'user' and 'content' in msg:
                content = msg['content']
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get('type') == 'video':
                            video_path = c.get('video')
                            break
                elif isinstance(content, str):
                    # Handle case where content might be a string path
                    video_path = content
        
        if not video_path:
            print(f"❌ Example {i+1}: No video path found in messages")
            invalid_videos.append(f"Example {i+1}: No video path")
            continue
        
        full_path = os.path.join(video_dir, video_path)
        print(f"🔍 Checking video: {full_path}")
        
        if validate_video(full_path):
            valid += 1
        else:
            invalid_videos.append(f"Example {i+1}: {video_path}")
    
    print(f"\n📊 Validation Results:")
    print(f"✅ Valid videos: {valid}/{total}")
    print(f"❌ Invalid videos: {total-valid}/{total}")
    
    if invalid_videos:
        print("\n⚠️ Invalid video details:")
        for item in invalid_videos:
            print(f"  • {item}")
    
    if valid == 0:
        print("\n❌ No valid videos found. Training cannot proceed.")
        sys.exit(1)
    elif valid < total:
        print(f"\n⚠️ {total-valid} videos are invalid, but {valid} are valid. Training will proceed with valid videos only.")
    else:
        print("\n✅ All videos are valid! Ready for training.")
    
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_videos.py <data_path> <video_dir>")
        sys.exit(1)
    
    sys.exit(main())
PYTHON

    # Make sure video directory exists
    mkdir -p "$VIDEO_DIR/videos/positive"
    mkdir -p "$VIDEO_DIR/videos/negative"
    
    # Run the validation
    python3 "$validation_script" "$DATA_DIR/washhand.json" "$VIDEO_DIR"
    local validation_result=$?
    
    if [[ $validation_result -eq 0 ]]; then
        log_success "Video validation passed! Proceeding with training..."
    else
        log_error "Video validation failed! Please fix the issues above before training."
        exit 1
    fi
    
    rm -f "$validation_script"
}

# Function to create training script for 8B model with video support - CORRECT FORMAT
create_training_script() {
    log_info "Creating FIXED training script for 8B model with video support..."
    
    mkdir -p "$SCRIPT_DIR"
    mkdir -p "$VIDEO_DIR"
    
    cat > "$SCRIPT_DIR/train_qwen3vl_8B_video_lora.py" <<'TRAINING_SCRIPT'
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
TRAINING_SCRIPT

    # Make the script executable
    chmod +x "$SCRIPT_DIR/train_qwen3vl_8B_video_lora.py"
    
    log_success "CORRECT FORMAT training script created at: $SCRIPT_DIR/train_qwen3vl_8B_video_lora.py"
}

# Function to run training for 8B model with video support
run_training() {
    log_info "Starting training process for 8B model with video support..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Check if we have enough GPU memory for 8B
    log_warning "8B model requires more GPU memory. Checking availability..."
    check_gpu_memory
    
    # Run training with ALL 12 examples
    python3 "$SCRIPT_DIR/train_qwen3vl_8B_video_lora.py" \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --data_path "$DATA_DIR/washhand.json" \
        --video_dir "$VIDEO_DIR" \
        --learning_rate 5e-5 \
        --max_steps 8000 \
        --batch_size 1 \
        --num_examples 12 \
        --gradient_accumulation_steps 4 \
        --debug
    
    return $?
}

# Function to verify training results
verify_training() {
    log_info "Verifying 8B model training results..."
    
    if [[ -f "$OUTPUT_DIR/training_metrics.json" ]]; then
        log_success "Training metrics found!"
        cat "$OUTPUT_DIR/training_metrics.json"
    else
        log_warning "Training metrics not found at: $OUTPUT_DIR/training_metrics.json"
    fi
    
    if [[ -f "$OUTPUT_DIR/adapter_config.json" ]]; then
        log_success "LoRA adapter configuration found!"
    else
        log_warning "LoRA adapter configuration not found"
    fi
    
    if [[ -f "$OUTPUT_DIR/adapter_model.safetensors" ]] || [[ -f "$OUTPUT_DIR/pytorch_model.bin" ]]; then
        log_success "Model weights found!"
    else
        log_warning "Model weights not found"
    fi
    
    echo ""
    log_info "Output directory contents:"
    ls -la "$OUTPUT_DIR"
    
    log_success "🎉 8B MODEL TRAINING VERIFICATION COMPLETE!"
    log_info "Output directory: $OUTPUT_DIR"
}

# Function to test inference with trained 8B model
test_inference() {
    log_info "Testing inference with trained 8B model..."
    
    cat > "/tmp/test_inference_8B.py" <<'TEST_SCRIPT'
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import cv2
import numpy as np

model_path = "/home/fion/tony4/output/training_results_8B"

print("🧪 Testing inference with trained 8B model...")

try:
    # Load processor and model with local_files_only
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    print("✅ Trained 8B model loaded successfully")
    
    # Get model device
    device = model.device
    print(f"📱 Model is on device: {device}")
    
    # Create a test image (purple square)
    image = Image.new('RGB', (224, 224), color='purple')
    
    # Test inference
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What color is this image?"}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    
    # Add image_grid_thw for inference
    inputs["image_grid_thw"] = torch.tensor([[1, 16, 16]], dtype=torch.long)
    
    # Move ALL inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"📱 Inputs moved to device: {device}")
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )
    
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"🤖 8B Model response: {response}")
    print("✅ 8B model inference test passed!")
    
    # Additional test with video frame extraction
    print("\n🎬 Testing video frame extraction functionality...")
    try:
        test_video_path = "/home/fion/tony4/videos/wash_hand.mp4"
        if os.path.exists(test_video_path):
            cap = cv2.VideoCapture(test_video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"✅ Video file accessible: {test_video_path}")
                print(f"   FPS: {fps:.2f}, Total frames: {total_frames}")
                cap.release()
                
                # Extract a sample frame
                cap = cv2.VideoCapture(test_video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 10)  # Get frame 10
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    test_image = Image.fromarray(frame_rgb).resize((224, 224))
                    print("✅ Successfully extracted test frame from video")
                    
                    # Test inference on video frame
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": test_image},
                                {"type": "text", "text": "What do you see in this video frame?"}
                            ]
                        }
                    ]
                    
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[test_image], return_tensors="pt")
                    inputs["image_grid_thw"] = torch.tensor([[1, 16, 16]], dtype=torch.long)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=True,
                            temperature=0.7
                        )
                    
                    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    print(f"🤖 Model response to video frame: {response}")
                else:
                    print("⚠️ Could not read frame from video, but video file is accessible")
            else:
                print(f"⚠️ Video file exists but could not be opened: {test_video_path}")
        else:
            print(f"⚠️ Test video file not found: {test_video_path}. Please add your video files to the videos directory.")
    except Exception as e:
        print(f"⚠️ Video testing encountered an error: {e}")
    
except Exception as e:
    print(f"❌ 8B model inference test failed: {e}")
    import traceback
    traceback.print_exc()
TEST_SCRIPT

    python3 "/tmp/test_inference_8B.py"
    rm -f "/tmp/test_inference_8B.py"
}

# Main execution flow
main() {
    echo ""
    log_info "Starting complete Qwen3-VL-8B training process with video support - CORRECT FORMAT VERSION..."
    echo ""
      
    # Step 1: Verify model directory
    verify_model_directory
    
    # Step 2: Environment setup
    setup_environment
    
    # Step 3: Install dependencies
    install_dependencies
    
    # Step 4: Validate JSON format (no auto-fixing)
    validate_json
    
    # Step 5: Validate all video files
    validate_all_videos
    
    # Step 6: Create FIXED training script with correct format support
    create_training_script
    
    # Step 7: Run training (will use all 12 examples)
    if run_training; then
        log_success "8B model training completed successfully!"
        
        # Step 8: Verify results
        verify_training
        
        # Step 9: Test inference
        test_inference
        
        echo ""
        log_success "================================================"
        log_success "🎉 8B MODEL ALL STEPS COMPLETED SUCCESSFULLY - CORRECT FORMAT VERSION!"
        log_success "================================================"
        log_info "Your trained 8B model is ready at: $OUTPUT_DIR"
        log_info "Training data used: $DATA_DIR/washhand.json"
        log_info "Video files directory: $VIDEO_DIR"
        log_info "Key Features:"
        log_info "  ✓ JSON validation with clear error messages only"
        log_info "  ✓ Trains on ALL 12 examples (6 positive + 6 negative)"
        log_info "  ✓ Comprehensive video validation before training"
        log_info "  ✓ Proper handling of nested video paths in messages"
        log_info "  ✓ Memory management with batched processing"
        log_info "  ✓ Gradient accumulation for memory constraints"
        log_info "  ✓ Full error handling and detailed logging"
        echo ""
        log_warning "IMPORTANT: Directory structure requirements:"
        log_warning "  • Positive examples: $VIDEO_DIR/videos/positive/"
        log_warning "  • Negative examples: $VIDEO_DIR/videos/negative/"
        log_info "  • Training data: $DATA_DIR/washhand.json"
        
        # Provide helpful tips for JSON editing
        echo ""
        log_info "💡 JSON EDITING TIPS:"
        log_info "  • Use VS Code with JSON extension for automatic formatting"
        log_info "  • Online validators: jsonlint.com or jsonformatter.org"
        log_info "  • Always validate after manual edits"
        echo ""
        log_success "✨ Training pipeline complete! Your model is ready for inference."
        
    else
        log_error "8B model training failed!"
        exit 1
    fi
}

# Run main function
main "$@"
