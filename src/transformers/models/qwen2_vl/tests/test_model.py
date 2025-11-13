import torch
import whisper
from transformers import Qwen2VLAudioConfig, Qwen2VLProcessor
from datasets import load_dataset
from qwen_vl_utils import fetch_audio
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2AudioTransformerPretrainedModel


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


print(" Testing Audio Processor Integration\n")

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
inputs = processor(
    text=[text],
    audios=[audio_array],
    return_tensors="pt",
    padding=False,
    truncation=False
)

print(inputs['input_ids'])
print(f"✓ Processed. Input keys: {list(inputs.keys())}\n")

# Step 7: Test audio model with audio_values from processor
print("Step 7: Testing audio model...")
config = Qwen2VLAudioConfig(whisper_model_name="turbo")
audio_model = Qwen2AudioTransformerPretrainedModel(config)

# Use audio_values from processor (already contains mel spectrograms)
audio_values = inputs['audio_values']  # Shape: [batch_size, n_mels, time] 
print(f"Audio values shape from processor: {audio_values.shape}") # torch.Size([1, 128, 3000])

# Pass to audio model (handles batched input automatically) Shape:[batch_size, time, embed_dim]  
output = audio_model(audio_values)
print(f"✓ Audio model output shape: {output.shape}") # torch.Size([1, 1500, 1280])