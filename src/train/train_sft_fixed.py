#!/usr/bin/env python3
"""
FIXED VERSION - Forces eager attention without breaking Hugging Face cache
"""
import os
import sys
import torch

# CRITICAL: Force eager attention BEFORE any transformers imports
os.environ['ATTN_IMPLEMENTATION'] = 'eager'
os.environ['USE_FLASH_ATTENTION'] = '0'
os.environ['FLASH_ATTENTION'] = 'OFF'
os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'

# Import transformers only after setting environment variables
import transformers
from transformers import AutoModelForCausalLM, AutoConfig

print("=== ENVIRONMENT SETUP ===")
print(f"ATTN_IMPLEMENTATION: {os.environ.get('ATTN_IMPLEMENTATION')}")
print(f"USE_FLASH_ATTENTION: {os.environ.get('USE_FLASH_ATTENTION')}")

# Safe import of the original training script
original_script_dir = os.path.dirname(os.path.abspath(__file__))
original_train_path = os.path.join(original_script_dir, 'train_sft.py')

# Read the original script
with open(original_train_path, 'r') as f:
    original_code = f.read()

# Execute the original script
exec(original_code)
