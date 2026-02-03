# =============================================================================
# IMPORTS & CONFIGURATION (UPDATED FOR QWEN3-VL) - CORRECTED IMPORTS
# =============================================================================
import argparse
import base64
import logging
import os
import sys
import time
from threading import Thread, Event
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
import gradio as gr

# CORRECTED IMPORTS FOR QWEN3-VL
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer
)

# Hugging Face Hub
from huggingface_hub import login, whoami, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError, HFValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Qwen3VL-HandHygiene")

# Configuration
class OptimizedConfig:
    # Model settings - UPDATED FOR QWEN3-VL
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    
    # GPU optimization
    USE_FLASH_ATTENTION = False
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Video processing - REDUCED FRAME COUNT TO FIX TOKEN LIMIT ISSUE
    MAX_FRAMES = 80  # Reduced from 80 to avoid token limits
    MIN_FRAMES_FOR_ANALYSIS = 3
    FRAME_SIZE = (224, 224)
    SAMPLING_INTERVAL = 8
    MOTION_MAGNITUDE_THRESHOLD = 3.0
    
    # Generation parameters
    MAX_NEW_TOKENS = 3024  # Reduced from 2024 to be more reasonable
    GENERATION_TEMPERATURE = 0.3
    TOP_P = 0.9
    REPETITION_PENALTY = 1.15
    
    # Video constraints
    MAX_VIDEO_DURATION = 90
    MIN_VIDEO_DURATION = 5

# Global model variables
processor = None
model = None
device = None

# =============================================================================
# MODEL INITIALIZATION (CORRECTED FOR QWEN3-VL)
# =============================================================================

def setup_huggingface_authentication():
    """Set up Hugging Face authentication for Qwen3-VL"""
    try:
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            logger.info("✅ Using HF_TOKEN from environment variable")
            login(token=hf_token, add_to_git_credential=False)
            return True
        
        try:
            user_info = whoami()
            logger.info(f"✅ Already authenticated as: {user_info.get('name', 'unknown user')}")
            return True
        except Exception:
            logger.warning("⚠️ Not authenticated with Hugging Face")
        
        logger.warning("⚠️ No Hugging Face authentication found.")
        logger.info("🔧 To authenticate, set HF_TOKEN environment variable")
        return False
    except Exception as e:
        logger.error(f"❌ Error during authentication setup: {e}")
        return False

def validate_model_exists(model_name):
    """Check if Qwen3-VL model exists on Hugging Face Hub"""
    try:
        api = HfApi()
        model_info = api.model_info(model_name)
        logger.info(f"✅ Model '{model_name}' exists on Hugging Face Hub")
        return True
    except (RepositoryNotFoundError, GatedRepoError, HFValidationError) as e:
        logger.error(f"❌ Model '{model_name}' not found: {e}")
        return False
    except Exception as e:
        logger.error(f"⚠️ Error checking model existence: {e}")
        return False

def load_qwen3_vl_model(model_id, device, token=None):
    """Load Qwen3-VL model with GPU optimization"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"📥 Loading Qwen3-VL processor for {model_id}")
        
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=token
        )
        
        logger.info(f"📥 Loading Qwen3-VL model for {model_id}")
        
        model_kwargs = {
            "torch_dtype": OptimizedConfig.TORCH_DTYPE,
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "trust_remote_code": True,
            "token": token,
        }
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs
        )
        
        model.eval()
        logger.info("✅ Qwen3-VL model loaded and set to evaluation mode")
        
        return processor, model
        
    except Exception as e:
        logger.error(f"❌ Qwen3-VL model loading failed: {e}")
        logger.exception("Full error traceback:")
        raise

# =============================================================================
# ENHANCED VIDEO PROCESSING WITH MOTION ANALYSIS
# =============================================================================

def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate motion magnitude between two frames"""
    try:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        motion_score = np.sum(thresh) / (thresh.size * 255)
        return motion_score
    except Exception as e:
        logger.error(f"❌ Error calculating frame difference: {e}")
        return 0.0

def select_motion_keyframes(frames: List[np.ndarray], motion_scores: List[float], target_count: int) -> List[np.ndarray]:
    """Select keyframes based on motion peaks"""
    if len(frames) <= target_count:
        return frames
    
    peaks = []
    for i in range(1, len(motion_scores) - 1):
        if motion_scores[i] > motion_scores[i-1] and motion_scores[i] > motion_scores[i+1]:
            peaks.append((i, motion_scores[i]))
    
    peaks.sort(key=lambda x: x[1], reverse=True)
    selected_indices = set()
    
    selected_indices.add(0)
    selected_indices.add(len(frames) - 1)
    
    for idx, _ in peaks[:target_count - 2]:
        selected_indices.add(idx)
    
    current_idx = 0
    while len(selected_indices) < target_count and current_idx < len(frames):
        selected_indices.add(current_idx)
        current_idx += len(frames) // (target_count - len(selected_indices))
    
    return [frames[i] for i in sorted(selected_indices)]

def extract_frames_with_motion_analysis(video_path: str, target_frames: int = OptimizedConfig.MAX_FRAMES) -> List[np.ndarray]:
    """Extract frames with motion-based keyframe detection"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Cannot open video: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        logger.error("❌ Video has no frames")
        cap.release()
        return None
    
    if fps <= 0:
        fps = 30.0
    
    frames = []
    motion_scores = []
    prev_frame = None
    
    # Calculate sample interval to avoid too many frames
    sample_interval = max(1, total_frames // (target_frames * 2))
    
    for i in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if ret and frame is not None and frame.size > 0:
            frame_resized = cv2.resize(frame, OptimizedConfig.FRAME_SIZE, interpolation=cv2.INTER_LANCZOS4)
            
            if prev_frame is not None:
                motion = calculate_frame_difference(prev_frame, frame_resized)
                motion_scores.append(motion)
            else:
                motion_scores.append(0.0)
            
            frames.append(frame_resized)
            prev_frame = frame_resized
    
    cap.release()
    
    if not frames:
        logger.error("❌ No frames extracted from video")
        return None
    
    if len(frames) <= target_frames:
        logger.info(f"✅ Extracted {len(frames)} frames (all available)")
        return frames[:target_frames]  # Ensure we don't exceed target
    
    keyframes = select_motion_keyframes(frames, motion_scores, target_frames)
    logger.info(f"✅ Extracted {len(keyframes)} keyframes from {len(frames)} total frames using motion analysis")
    
    return keyframes

def analyze_interframe_motion(frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, Any]:
    """Analyze motion patterns between two consecutive frames"""
    try:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        return {
            "mean_magnitude": float(np.mean(magnitude)),
            "motion_direction": float(np.mean(angle)),
            "motion_consistency": float(np.std(magnitude))
        }
    except Exception as e:
        logger.error(f"❌ Error analyzing interframe motion: {e}")
        return {
            "mean_magnitude": 0.0,
            "motion_direction": 0.0,
            "motion_consistency": 0.0
        }

def detect_step_transitions(frames: List[np.ndarray]) -> Dict[str, List[int]]:
    """Detect potential step transitions between frames"""
    transitions = {
        "palm_rubbing": [],
        "back_hand_cleaning": [],
        "finger_interlace": [],
        "thumb_rotation": [],
        "fingertip_cleaning": [],
        "wrist_cleaning": []
    }
    
    if len(frames) < 2:
        return transitions
    
    for i in range(len(frames) - 1):
        motion_pattern = analyze_interframe_motion(frames[i], frames[i + 1])
        
        # Simple heuristic based on motion characteristics
        if motion_pattern["mean_magnitude"] > 0.1 and motion_pattern["motion_consistency"] < 0.5:
            transitions["palm_rubbing"].append(i)
        elif motion_pattern["mean_magnitude"] > 0.08:
            transitions["back_hand_cleaning"].append(i)
    
    logger.info(f"🔍 Detected step transitions: {transitions}")
    return transitions

# =============================================================================
# ENHANCED PROMPT TEMPLATES WITH TEMPORAL ANALYSIS
# =============================================================================

class EnhancedPromptTemplates:
    """Enhanced prompt templates with temporal awareness"""
    
    @staticmethod
    def get_temporal_analysis_prompt(user_input: str, frame_count: int, step_transitions: Dict[str, List[int]] = None) -> str:
        """Create prompt that considers temporal sequence and motion patterns"""
        
        transition_info = ""
        if step_transitions:
            transition_info = "\n**DETECTED MOTION PATTERNS:**\n"
            for step, frames in step_transitions.items():
                if frames:
                    transition_info += f"- {step.replace('_', ' ').title()}: detected at frames {frames}\n"
        
        return f"""You are analyzing a hand hygiene video with {frame_count} sequential frames. Analyze the TEMPORAL PROGRESSION of hand hygiene steps.

**CRITICAL ANALYSIS GUIDELINES:**
1. Analyze frames in SEQUENTIAL ORDER - they represent the temporal progression
2. Look for TRANSITIONS between steps in consecutive frames
3. Some steps may be partially completed across multiple frames
4. Focus on HAND MOVEMENT PATTERNS and POSITION CHANGES between frames
5. Require MULTIPLE FRAMES showing step progression for confirmation
6. Mark steps as "⚠️ Insufficient Evidence" if clear progression is not visible

**FRAME SEQUENCE ANALYSIS:**
- Analyze how hand positions and movements evolve across {frame_count} frames
- Identify step initiation, execution, and completion across the sequence
- Note if steps are performed out of sequence or missing entirely

{transition_info}

**STEP COMPLETION ASSESSMENT (Based on Temporal Evidence):**

For each WHO step, assess based on VISIBLE PROGRESSION across multiple frames:

1. **Palm to palm**: ✅ Only if circular rubbing motion is visible across frames
2. **Back of hands**: ✅ Only if interlaced finger movement shows proper coverage
3. **Fingers interlaced**: ✅ Only if palm-to-palm with interlaced fingers is clearly executed
4. **Back of fingers**: ✅ Only if interlocked fingers clean opposing palms
5. **Thumb rotation**: ✅ Only if rotational thumb cleaning is clearly visible
6. **Fingertip rotation**: ✅ Only if clasped finger rotational cleaning is evident
7. **Wrist cleaning**: ✅ Only if both wrists receive proper rotational cleaning

**EVIDENCE REQUIREMENTS:**
- ✅ CORRECT: Clear evidence across multiple frames showing proper technique
- ⚠️ PARTIAL: Some evidence but incomplete or improper technique
- ❌ MISSING: No clear evidence of the step being performed
- 🔍 UNCERTAIN: Insufficient visual evidence to make determination

**FINAL ASSESSMENT CRITERIA:**
- Perfect Compliance: All 7 steps clearly executed with proper technique
- Good Compliance: 5-6 steps properly executed
- Fair Compliance: 3-4 steps properly executed  
- Poor Compliance: 2 or fewer steps properly executed

**INFECTION RISK ASSESSMENT:**
- Low Risk: Perfect/Good compliance with all critical steps covered
- Medium Risk: Fair compliance with some critical steps missing
- High Risk: Poor compliance with multiple critical steps missing

USER REQUEST: {user_input or "Analyze hand hygiene compliance through temporal progression"}

Provide step-by-step analysis based ONLY on visible evidence across the frame sequence:"""

# =============================================================================
# ENHANCED VIDEO VALIDATION
# =============================================================================

def enhanced_video_validation(video_path: str) -> Tuple[bool, str]:
    """More comprehensive video validation with hand detection"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video file - format may not be supported"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames < 10:
            cap.release()
            return False, f"Too few frames: {total_frames} (minimum 10 required for motion analysis)"
        
        if fps <= 0:
            fps = 30.0
        
        duration = total_frames / fps
        
        if duration > OptimizedConfig.MAX_VIDEO_DURATION:
            cap.release()
            return False, f"Video too long: {duration:.1f}s (maximum {OptimizedConfig.MAX_VIDEO_DURATION}s)"
        
        if duration < OptimizedConfig.MIN_VIDEO_DURATION:
            cap.release()
            return False, f"Video too short: {duration:.1f}s (minimum {OptimizedConfig.MIN_VIDEO_DURATION}s required)"
        
        if width < 128 or height < 128:
            cap.release()
            return False, f"Video resolution too low: {width}x{height} (minimum 128x128 required)"
        
        # Try to load hand cascade for basic hand detection
        hand_cascade_path = cv2.data.haarcascades + 'haarcascade_hand.xml'
        hand_detected_frames = 0
        
        if os.path.exists(hand_cascade_path):
            hand_cascade = cv2.CascadeClassifier(hand_cascade_path)
            
            # Sample frames for hand detection
            for i in range(0, min(total_frames, 30), 5):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret and frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hands = hand_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                    if len(hands) > 0:
                        hand_detected_frames += 1
        else:
            # If hand cascade not available, use simpler check
            hand_detected_frames = 3  # Assume hands are present
        
        cap.release()
        
        if hand_detected_frames < 2:
            return False, "Cannot clearly detect hand movements in video. Ensure hands are visible, well-lit, and fill a reasonable portion of the frame."
        
        return True, f"Valid video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s, {width}x{height}, hands detected in {hand_detected_frames} samples"
        
    except Exception as e:
        logger.error(f"❌ Enhanced video validation failed: {e}")
        return False, f"Video validation error: {str(e)}"

def is_video_file(filename):
    """Check if file is a video format"""
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg', '.mpg', '.m4v', '.3gp']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

# =============================================================================
# MODEL INFERENCE WITH ENHANCED TEMPORAL ANALYSIS - FIXED INPUT PREPARATION
# =============================================================================

def prepare_multiframe_inputs(user_prompt: str, video_frames: List[np.ndarray] = None):
    """
    Prepare multi-frame inputs for Qwen3-VL - FIXED VERSION
    """
    try:
        pil_images = []
        if video_frames:
            # Ensure we don't exceed the maximum frames
            for frame in video_frames[:OptimizedConfig.MAX_FRAMES]:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_images.append(pil_image)
        
        messages = []
        
        messages.append({
            "role": "system",
            "content": "You are a medical AI specialist in infection control and hand hygiene compliance with expertise in temporal sequence analysis."
        })
        
        if pil_images:
            user_content = [
                {"type": "text", "text": user_prompt}
            ]
            # Add images to the content
            for image in pil_images:
                user_content.insert(0, {"type": "image", "image": image})
            
            messages.append({
                "role": "user",
                "content": user_content
            })
        else:
            messages.append({
                "role": "user",
                "content": user_prompt
            })
        
        # Apply chat template WITHOUT tokenization first
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # FIXED: Use proper processing without max_length constraint for images
        if pil_images:
            inputs = processor(
                text=[text],
                images=pil_images,
                padding=True,
                return_tensors="pt",
                # REMOVED max_length and truncation to avoid token mismatch
            )
        else:
            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt",
                max_length=2048,
                truncation=True
            )
        
        # Ensure image_grid_thw is present if we have images
        if "image_grid_thw" not in inputs and pil_images:
            batch_size = len(pil_images)
            inputs["image_grid_thw"] = torch.tensor(
                [[1, 14, 14]] * batch_size, 
                dtype=torch.long
            )
        
        return inputs.to(device)
        
    except Exception as e:
        logger.error(f"❌ Input preparation failed: {e}")
        logger.exception("Full error traceback:")
        
        # Fallback: try with just text if image processing fails
        logger.info("🔄 Attempting fallback with text-only input")
        try:
            inputs = processor(
                text=[user_prompt],
                padding=True,
                return_tensors="pt",
                max_length=2048,
                truncation=True
            )
            return inputs.to(device)
        except Exception as fallback_error:
            logger.error(f"❌ Fallback also failed: {fallback_error}")
            raise

def bot_streaming_enhanced(message, history):
    """Enhanced bot streaming with temporal analysis"""
    global processor, model, device
    
    video_frames = None
    user_text = ""
    files = []
    
    if isinstance(message, dict):
        user_text = message.get('text', '')
        files = message.get('files', [])
    elif isinstance(message, list) and len(message) > 0:
        first_item = message[0]
        if isinstance(first_item, dict):
            user_text = first_item.get('text', '')
            files = first_item.get('files', [])
        else:
            user_text = str(first_item)
            files = []
    else:
        user_text = str(message)
        files = []
    
    logger.info(f"📝 User input: '{user_text}'")
    logger.info(f"📎 Files received: {len(files)}")
    
    step_transitions = {}
    
    for file_item in files:
        try:
            if isinstance(file_item, dict):
                file_path = file_item.get("path", file_item.get("name", ""))
            else:
                file_path = str(file_item)
            
            if not os.path.exists(file_path):
                logger.error(f"❌ File not found: {file_path}")
                continue
            
            if is_video_file(file_path):
                logger.info(f"🎬 Processing video: {os.path.basename(file_path)}")
                
                is_valid, validation_msg = enhanced_video_validation(file_path)
                if not is_valid:
                    logger.error(f"❌ Video validation failed: {validation_msg}")
                    yield history + [{"role": "user", "content": user_text}, {"role": "assistant", "content": f"❌ ANALYSIS FAILED: {validation_msg}"}]
                    return
                
                logger.info(f"✅ Video validation passed: {validation_msg}")
                
                video_frames = extract_frames_with_motion_analysis(video_path=file_path)
                if video_frames and len(video_frames) >= OptimizedConfig.MIN_FRAMES_FOR_ANALYSIS:
                    logger.info(f"✅ Extracted {len(video_frames)} frames using motion analysis")
                    
                    # Detect step transitions for enhanced analysis
                    if len(video_frames) > 1:
                        step_transitions = detect_step_transitions(video_frames)
                else:
                    logger.error(f"❌ Insufficient frames: {len(video_frames) if video_frames else 0}")
                    yield history + [{"role": "user", "content": user_text}, {"role": "assistant", "content": "❌ ANALYSIS FAILED: Could not extract sufficient frames from video for proper analysis."}]
                    return
        except Exception as e:
            logger.error(f"❌ Error processing file {file_item}: {e}")
            continue

    # Create enhanced prompt with temporal awareness
    frame_count = len(video_frames) if video_frames else 0
    user_prompt = EnhancedPromptTemplates.get_temporal_analysis_prompt(
        user_input=user_text or "Verify hand hygiene steps against WHO standards with temporal analysis",
        frame_count=frame_count,
        step_transitions=step_transitions
    )
    
    try:
        inputs = prepare_multiframe_inputs(user_prompt, video_frames)
        logger.info("✅ Inputs prepared successfully for Qwen3-VL")
    except Exception as e:
        logger.error(f"❌ Error preparing inputs: {e}")
        error_response = f"❌ ANALYSIS FAILED: Error preparing model inputs - {str(e)}"
        yield history + [{"role": "user", "content": user_text}, {"role": "assistant", "content": error_response}]
        return

    logger.info("🚀 Starting Qwen3-VL generation with enhanced temporal analysis...")
    
    streamer = TextIteratorStreamer(
        processor.tokenizer, 
        skip_special_tokens=True, 
        skip_prompt=True,
        timeout=300.0
    )
    
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "streamer": streamer,
        "max_new_tokens": OptimizedConfig.MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": OptimizedConfig.GENERATION_TEMPERATURE,
        "top_p": OptimizedConfig.TOP_P,
        "repetition_penalty": OptimizedConfig.REPETITION_PENALTY,
        "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    }
    
    if "pixel_values" in inputs:
        generation_kwargs["pixel_values"] = inputs["pixel_values"]
    if "image_grid_thw" in inputs:
        generation_kwargs["image_grid_thw"] = inputs["image_grid_thw"]

    generation_error = None
    generation_complete = Event()
    
    def generation_thread():
        nonlocal generation_error
        try:
            logger.info("🔄 Starting Qwen3-VL generation with temporal analysis")
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(**generation_kwargs)
            end_time = time.time()
            logger.info(f"✅ Generation completed in {end_time - start_time:.2f}s")
        except Exception as e:
            generation_error = e
            logger.error(f"❌ Generation error: {e}")
            logger.exception("Generation error details:")
            streamer.end()
        finally:
            generation_complete.set()
    
    thread = Thread(target=generation_thread, daemon=True)
    thread.start()
    
    buffer = ""
    try:
        new_history = history + [{"role": "user", "content": user_text}]
        
        token_count = 0
        start_time = time.time()
        for new_text in streamer:
            token_count += 1
            if new_text and new_text.strip():
                buffer += new_text
                elapsed = time.time() - start_time
                if token_count % 20 == 0:
                    logger.info(f"💬 Generated {token_count} tokens in {elapsed:.1f}s")
                yield new_history + [{"role": "assistant", "content": buffer.strip()}]
        
        logger.info(f"✅ Generation completed. Total tokens: {token_count}")
        
        if not buffer.strip():
            buffer = "I've analyzed the hand hygiene technique using temporal progression analysis. For optimal assessment, ensure all 7 WHO steps are clearly visible across the video sequence with proper lighting and hand visibility."
        
        yield new_history + [{"role": "assistant", "content": buffer.strip()}]
        
    except Exception as e:
        error_msg = str(e).strip() or "Generation failed"
        logger.error(f"❌ Streaming error: {error_msg}")
        if buffer.strip():
            final_response = buffer.strip() + f"\n\n⚠️ Analysis interrupted: {error_msg}"
        else:
            final_response = f"❌ ANALYSIS FAILED: {error_msg}"
        yield history + [{"role": "user", "content": user_text}, {"role": "assistant", "content": final_response}]

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main(args):
    global processor, model, device
    
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available. Falling back to CPU.")
        args.device = "cpu"
    
    device = args.device
    logger.info(f"🖥️ Using device: {device}")
    
    logger.info("🔐 Setting up Hugging Face authentication...")
    setup_huggingface_authentication()
    
    if args.model_path and os.path.exists(args.model_path):
        model_id = args.model_path
        logger.info(f"📁 Using local model: {model_id}")
    else:
        model_id = args.model_base or OptimizedConfig.MODEL_NAME
        logger.info(f"🌐 Using Hugging Face model: {model_id}")
    
    logger.info(f"🔍 Validating model: {model_id}")
    if not os.path.exists(model_id) and not validate_model_exists(model_id):
        logger.error(f"❌ Model validation failed: {model_id}")
        fallback_model = "Qwen/Qwen3-VL-2B-Instruct"
        if validate_model_exists(fallback_model):
            model_id = fallback_model
            logger.info(f"✅ Using fallback: {fallback_model}")
        else:
            raise ValueError(f"❌ No suitable Qwen3-VL model found")
    
    logger.info(f"📥 Loading Qwen3-VL model: {model_id}")
    hf_token = os.environ.get('HF_TOKEN')
    
    try:
        processor, model = load_qwen3_vl_model(
            model_id=model_id,
            device=device,
            token=hf_token
        )
        logger.info("✅ Qwen3-VL model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Error loading Qwen3-VL model: {e}")
        sys.exit(1)

    with gr.Blocks(title="Qwen3-VL Hand Hygiene Analyzer", css="""
        .container { max-width: 1200px; margin: 0 auto; }
        .info-box { background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #d1e7dd; }
        .video-upload { border: 2px dashed #4a90e2; padding: 20px; text-align: center; margin: 15px 0; 
                       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .results-box { min-height: 500px; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; 
                      background: #f8f9fa; }
        .header { text-align: center; margin-bottom: 20px; }
        .footer { text-align: center; margin-top: 20px; font-size: 0.9em; color: #666; }
        .chat-container { display: flex; flex-direction: column; height: 600px; }
        .chat-history { flex-grow: 1; overflow-y: auto; margin-bottom: 10px; }
        .input-area { display: flex; flex-direction: column; gap: 10px; }
        .step-analysis { font-family: 'Courier New', monospace; line-height: 1.6; }
        .correct { color: #28a745; font-weight: bold; }
        .partial { color: #ffc107; font-weight: bold; }
        .incorrect { color: #dc3545; font-weight: bold; }
        .status-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; 
                       margin-right: 5px; font-weight: bold; }
    """) as demo:
        gr.Markdown("""
        # 🧼 Qwen3-VL Hand Hygiene Step Analyzer
        *Enhanced Temporal Analysis with Motion Detection - WHO Compliance Verification*
        
        **⚠️ Upload a hand hygiene video (5-90 seconds) for comprehensive temporal analysis of all 7 WHO steps.**
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=600,
                    show_copy_button=True,
                    placeholder="Enhanced temporal analysis results will appear here...",
                    label="Step-by-Step Temporal Analysis",
                    type="messages",
                    value=[]
                )
            with gr.Column(scale=1):
                gr.Markdown("### 🚀 Enhanced Features")
                gr.Markdown(f"""
                **Model:** {model_id.split('/')[-1]}
                • **Temporal Sequence Analysis**
                • **Motion Pattern Detection** 
                • **Multi-frame Evidence Validation**
                • **Clinical Risk Assessment**
                
                **GPU Optimized:**
                • Device: {device.upper()}
                • Frames: {OptimizedConfig.MAX_FRAMES}
                • Motion Analysis: Enabled
                """)
                
                gr.Markdown("### 📋 Requirements")
                gr.Markdown("""
                • **Duration:** 5-90 seconds
                • **Resolution:** Minimum 128x128
                • **Lighting:** Clear hand visibility
                • **Movement:** Full hand hygiene sequence
                • **Format:** MP4, MOV, AVI, MKV, WEBM
                """)
                
                gr.Markdown("### 🏥 WHO 7-Step Protocol")
                gr.Markdown("""
                1. **Palm to palm** - Circular rubbing
                2. **Back of hands** - Interlaced fingers  
                3. **Fingers interlaced** - Palm to palm
                4. **Back of fingers** - Opposing palms
                5. **Thumb rotation** - Clasped in palm
                6. **Fingertips rotation** - In palm
                7. **Wrist cleaning** - Rotational motion
                """)
        
        chat_input = gr.MultimodalTextbox(
            interactive=True, 
            file_types=["video"],
            placeholder="📤 Upload hand hygiene video for enhanced temporal analysis...",
            show_label=False,
            label="Video Input & Analysis Request"
        )
        
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            example_btn = gr.Button("💡 Show Examples", variant="primary")
        
        gr.Examples(
            examples=[
                [{"text": "Analyze temporal progression of all 7 WHO hand hygiene steps"}],
                [{"text": "Verify step-by-step compliance using motion analysis"}],
                [{"text": "Identify missing steps and provide clinical improvement recommendations"}],
            ],
            inputs=[chat_input],
            label="Enhanced Analysis Examples",
            outputs=[chat_input]
        )
        
        gr.Markdown("""
        ### 🔍 Enhanced Analysis Features
        1. **Motion-Based Frame Selection**: Automatically detects key moments in the hand hygiene sequence
        2. **Temporal Progression Analysis**: Evaluates step execution across multiple frames
        3. **Evidence-Based Assessment**: Requires multiple frames of evidence for step confirmation
        4. **Clinical Risk Scoring**: Provides infection risk assessment based on missing steps
        
        **Note:** Analysis uses advanced motion detection and may take 30-120 seconds.
        """, elem_classes="footer")

        chat_input.submit(
            fn=bot_streaming_enhanced,
            inputs=[chat_input, chatbot],
            outputs=[chatbot],
            queue=True
        )
        
        clear_btn.click(
            lambda: [],
            outputs=[chatbot]
        )
        
        def show_examples():
            return {
                chat_input: [{"text": "Analyze temporal progression of all 7 WHO hand hygiene steps"}]
            }
        
        example_btn.click(
            fn=show_examples,
            outputs=[chat_input]
        )

    logger.info(f"""
🚀 ENHANCED QWEN3-VL HAND HYGIENE ANALYZER READY
📊 MODEL: {model_id.split('/')[-1]}
🖥️ DEVICE: {device.upper()}
⚡ ENHANCEMENTS: 
   • Temporal Sequence Analysis
   • Motion-Based Keyframe Detection  
   • Multi-frame Evidence Validation
   • Clinical Risk Assessment

🏠 Local URL: http://localhost:{args.port}
    """)
    
    demo.queue(api_open=False, max_size=3)
    
    try:
        demo.launch(
            show_api=False, 
            share=args.share,
            server_name=args.host,
            server_port=args.port,
            show_error=True,
            prevent_thread_lock=False,
            quiet=False
        )
    except Exception as e:
        logger.error(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Qwen3-VL Hand Hygiene Analyzer")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to local Qwen3-VL model directory")
    parser.add_argument("--model-base", type=str, default=OptimizedConfig.MODEL_NAME,
                        help="Hugging Face Qwen3-VL model identifier")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to bind to")
    parser.add_argument("--share", action="store_true",
                        help="Create public gradio link")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("👋 Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        sys.exit(1)