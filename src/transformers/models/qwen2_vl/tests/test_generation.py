import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Qwen2VLConfig
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


print(" Testing Audio Generation End-to-End\n")

# Step 1: Load processor
print("Step 1: Loading processor...")
processor = Qwen2VLProcessor.from_pretrained("moutasem/qwen2-vl-7b-audio")
print("✓ Processor loaded\n")

# Step 2: Load model with audio support enabled
print("Step 2: Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure audio encoder
config = Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
config.audio_config.whisper_model_name = 'turbo'  # Enable Whisper encoder

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    config=config,
    dtype=torch.float16 if device == "cuda" else torch.float32
).to(device).eval()
print(f"✓ Model loaded on {device} with audio support enabled\n")

# Step 3: Load sample from dataset
print("Step 3: Loading audio sample from dataset...")
dataset = load_dataset("speechbrain/LargeScaleASR", "small", split="train", streaming=True)
sample = next(iter(dataset.take(1)))
print("✓ Sample loaded\n")

# Step 4: Extract audio array
print("Step 4: Extracting audio...")
audio_array, sample_rate = fetch_audio(sample, sample_rate=16000)
print(f"✓ Audio shape: {audio_array.shape}, Sample rate: {sample_rate}Hz\n")

# Step 5: Format as ChatML
print("Step 5: Formatting as ChatML...")
messages = format_as_chatml(sample, include_assistant=False)
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("✓ Template applied\n")

# Step 6: Process inputs
print("Step 6: Processing text + audio...")
print("  This step processes audio into mel spectrogram (may take a moment)...")
inputs = processor(
    text=[text],
    audios=[audio_array],
    return_tensors="pt",
    padding=False,
    truncation=False
).to(device)
print(f"✓ Processed. Input keys: {list(inputs.keys())}")
if 'audio_values' in inputs:
    print(f"  Audio values shape: {inputs['audio_values'].shape}")
if 'audio_grid_thw' in inputs:
    print(f"  Audio grid_thw: {inputs['audio_grid_thw']}")
print(f"  Input IDs shape: {inputs['input_ids'].shape}\n")

# Step 7: Generate transcription
print("Step 7: Generating transcription...")
print(f"  Input sequence length: {inputs['input_ids'].shape[1]}")
print(f"  This may take a while with long audio sequences...")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,  # Reduced for faster testing
        do_sample=False,  # Use greedy decoding for faster generation
        # do_sample=True,  # Uncomment for sampling (slower)
        # temperature=0.7,  
        # top_p=0.9,
        # repetition_penalty=1.2
    )
print("✓ Generation complete\n")
print(f"Generated IDs shape: {generated_ids.shape}")

# Step 8: Decode output
print("Step 8: Decoding output...")
generated_text = processor.batch_decode(
    generated_ids, # torch.Size([1, 3156])
    skip_special_tokens=True, # true means skip the special tokens like <|startoftext|> and <|endoftext|>
    clean_up_tokenization_spaces=False
)[0]

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Ground Truth: {sample['text']}")
print(f"Generated:    {generated_text}")
print("="*50 + "\n")

print("✓ Test completed successfully!")

