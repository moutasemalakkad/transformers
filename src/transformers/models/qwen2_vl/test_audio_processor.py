# """
# Test the modified Qwen2-VL processor with audio support
# """
# from transformers import Qwen2VLProcessor, AutoTokenizer
# from datasets import load_dataset
# import numpy as np

# print("🧪 Testing Audio Processor Integration\n")

# # Step 1: Load the BASE processor (with your local modifications)
# print("Step 1: Loading base processor (with local audio modifications)...")
# processor = Qwen2VLProcessor.from_pretrained("moutasem/qwen2-vl-7b-audio")

# # Step 2: Replace tokenizer with your custom one that has audio tokens
# print("Step 2: Loading custom tokenizer with audio tokens...")
# processor.tokenizer = AutoTokenizer.from_pretrained("moutasem/qwen2-vl-7b-audio")
# print("✓ Processor loaded with audio-enabled tokenizer\n")

# # Step 3: Load a sample from your dataset
# print("Step 3: Loading audio sample from dataset...")
# dataset = load_dataset("speechbrain/LargeScaleASR", "small", split="train", streaming=True)
# sample = next(iter(dataset.take(1)))

# # Step 4: Get audio array using your fetch_audio function
# print("Step 4: Processing audio with fetch_audio...")
# from qwen_vl_utils.audio_process import fetch_audio

# audio_array, sample_rate = fetch_audio(sample["wav"], sample_rate=16000)
# print(f"✓ Audio shape: {audio_array.shape}, Sample rate: {sample_rate}Hz\n")

# # Step 5: Format the conversation
# print("Step 5: Formatting conversation with chat template...")


# messages = [
#     {
#         "role": "system",
#         "content": [
#             {"type": "text", "text": "You are an automatic speech recognition assistant."}
#         ]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "audio"},  # Just the marker
#             {"type": "text", "text": "Please transcribe this clip."}
#         ]
#     }
# ]

# # Apply chat template to get formatted text
# text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
# print(f"✓ Formatted text:\n{text}\n")

# # Step 6: Process with audio
# print("Step 6: Processing text + audio through modified processor...")
# try:
#     print(f"DEBUG: Tokenizer max_length: {processor.tokenizer.model_max_length}")
#     print(f"DEBUG: Text length before tokenization: {len(text[0])}")
#     inputs = processor(
#         text=[text],
#         audios=[audio_array],  # Pass audio separately
#         return_tensors="pt",
#         # truncation=False,
#         # padding=False,
#     )
    
#     print("✓ Processor output keys:", list(inputs.keys()))
#     print(f"✓ Input IDs shape: {inputs['input_ids'].shape}")
    
#     if 'audio_values' in inputs:
#         print(f"✓ Audio values shape: {inputs['audio_values'].shape}")
#         print(f"✓ Audio grid thw: {inputs['audio_grid_thw']}")
#         print("\n🎉 SUCCESS! Audio processing is working!\n")
#     else:
#         print("\n⚠️  WARNING: 'audio_values' not in outputs!")
#         print("Available keys:", list(inputs.keys()))
    
#     # Verify audio tokens are in the input
#     audio_pad_id = processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
#     num_audio_tokens = (inputs['input_ids'] == audio_pad_id).sum().item()
#     print(f"✓ Found {num_audio_tokens} audio pad tokens in input")
#     print(f"  (Should be ~1500 for 30-second audio)")
    
# except Exception as e:
#     print(f"❌ Error: {e}")
#     import traceback
#     traceback.print_exc()



"""
Test the modified Qwen2-VL processor with audio support
"""
from transformers import Qwen2VLProcessor
from datasets import load_dataset
import sys
import os

# Add utils to path
sys.path.insert(0, '/Users/moutasem/code/fine_tuning_vlm_for_speech_understanding')
from utils import to_chatml, extract_audio_array
from utils import to_chatml, extract_audio_array

print("Testing Audio Processor Integration\n")

# Step 1: Load processor
print("Step 1: Loading processor with audio tokens...")
processor = Qwen2VLProcessor.from_pretrained("moutasem/qwen2-vl-7b-audio")
print("Processor loaded\n")

# Step 2: Load sample from dataset
print("Step 2: Loading audio sample from dataset...")
dataset = load_dataset("speechbrain/LargeScaleASR", "small", split="train", streaming=True)
sample = next(iter(dataset.take(1)))

# Step 3: Extract audio array
print("Step 3: Extracting audio...")
audio_array = extract_audio_array(sample)
print(f"Audio shape: {audio_array.shape}\n")

# Step 4: Format as ChatML
print("Step 4: Formatting as ChatML...")
messages = to_chatml(sample, include_assistant=False)  # No assistant for inference
print(f"Messages: {len(messages)} messages\n")

# Step 5: Apply chat template
print("Step 5: Applying chat template...")
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Formatted text:\n{text}\n")

# Step 6: Process with audio
print("Step 6: Processing through processor...")
try:
    inputs = processor(
        text=[text],
        audios=[audio_array],
        return_tensors="pt",
        padding=False,
        truncation=False
    )
    
    print("Processor output keys:", list(inputs.keys()))
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    
    if 'audio_values' in inputs:
        print(f"Audio values shape: {inputs['audio_values'].shape}")
        print(f"Audio grid thw: {inputs['audio_grid_thw']}")
        print("\nSUCCESS! Audio processing is working!\n")
    else:
        print("\nWARNING: 'audio_values' not in outputs!")
    
    # Verify audio tokens
    audio_pad_id = processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    num_audio_tokens = (inputs['input_ids'] == audio_pad_id).sum().item()
    print(f"Found {num_audio_tokens} audio pad tokens in input")
    
    # Show what the expected transcription is
    print(f"\nExpected transcription: {sample['text']}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()