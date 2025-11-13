import torch
import whisper
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig

# Load standalone Whisper # check transfomer layers weights
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder_standalone = whisper.load_model('turbo', device=device).encoder

print(encoder_standalone)

# Load Whisper via Qwen2VL from_pretrained
config = Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
config.audio_config.whisper_model_name = 'turbo'
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    config=config,
    dtype=torch.float16 if device == "cuda" else torch.float32
).to(device).eval()


encoder_qwen = model.model.audio.model.encoder

# Compare conv1 layer
weights_match = torch.allclose(encoder_standalone.conv1.weight, encoder_qwen.conv1.weight, atol=1e-5)
weights_match_att = torch.allclose(encoder_standalone.blocks[0].attn.query.weight, encoder_qwen.blocks[0].attn.query.weight, atol=1e-5)

print(f"Conv1 weight shape: {encoder_standalone.conv1.weight.shape}")
print(f"Weights match: {weights_match}")
print(f"Weights match att: {weights_match_att}")


