

"""
Test the modified Qwen2-VL processor with audio support
"""
from transformers import Qwen2VLProcessor
from datasets import load_dataset
from qwen_vl_utils import fetch_audio


def format_as_chatml(sample, include_assistant=True):
    """
    Convert dataset sample to ChatML format for the processor.
    
    Args:
        sample: Dataset sample with 'wav' and 'text' keys
        include_assistant: Whether to include the assistant's response (for training)
    
    Returns:
        List of message dictionaries
    """
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an automatic speech recognition assistant."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio"},  # Marker for audio position
                {"type": "text", "text": "Transcribe this audio."}
            ]
        }
    ]
    
    if include_assistant:
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["text"]}
            ]
        })
    
    return messages


print("🧪 Testing Audio Processor Integration\n")

# Step 1: Load processor
print("Step 1: Loading processor with audio tokens...")
processor = Qwen2VLProcessor.from_pretrained("moutasem/qwen2-vl-7b-audio")
print("✓ Processor loaded\n")

# Step 2: Load sample from dataset
print("Step 2: Loading audio sample from dataset...")
dataset = load_dataset("speechbrain/LargeScaleASR", "small", split="train", streaming=True)
sample = next(iter(dataset.take(1)))
print("✓ Sample loaded\n")

# Step 3: Extract audio array using fetch_audio
print("Step 3: Extracting audio with fetch_audio...")
audio_array, sample_rate = fetch_audio(sample, sample_rate=16000)
print(f"✓ Audio shape: {audio_array.shape}, Sample rate: {sample_rate}Hz\n")

# Step 4: Format as ChatML
print("Step 4: Formatting as ChatML...")
messages = format_as_chatml(sample, include_assistant=False)
print(f"✓ Formatted {len(messages)} messages\n")

# Step 5: Apply chat template
print("Step 5: Applying chat template...")
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"✓ Template applied\n")

# Step 6: Process with audio
print("Step 6: Processing text + audio through processor...")
try:
    inputs = processor(
        text=[text],
        audios=[audio_array],
        return_tensors="pt",
        padding=False,
        truncation=False
    )
    
    print("✓ Processor output keys:", list(inputs.keys()))
    print(f"✓ Input IDs shape: {inputs['input_ids'].shape}")
    
    if 'audio_values' in inputs:
        print(f"✓ Audio values shape: {inputs['audio_values'].shape}")
        print(f"✓ Audio grid thw: {inputs['audio_grid_thw']}")
        
        # Verify audio tokens
        audio_pad_id = processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        num_audio_tokens = (inputs['input_ids'] == audio_pad_id).sum().item()
        print(f"✓ Found {num_audio_tokens} audio pad tokens in input")
        print(f"  (Expected ~1500 for 30-second audio)")
        
        print(f"\n✓ Expected transcription: {sample['text']}")
        print("\n🎉 SUCCESS! Audio processing is working!\n")
    else:
        print("\n⚠️  WARNING: 'audio_values' not in outputs!")
        print("Available keys:", list(inputs.keys()))
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()