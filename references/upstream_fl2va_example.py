import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video

# 61.7GB of transformer and 62.1GB of conditioner do not sit on one accelerator, so the components are
# registered in a manager that moves each one on and off as the blocks reach it. See [Memory](#memory).
manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")

pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", components_manager=manager)
pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)

prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
# `output=` returns exactly the named outputs instead of the whole pipeline state.
outputs = ["videos", "audio", "sampling_rate"]

# Text to video + audio.
results = pipe(prompt=prompt, num_frames=124, generator=torch.Generator().manual_seed(42), output=outputs)

# First frame (and optionally last frame) to video + audio. The canvas follows the first keyframe.
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
results = pipe(prompt=prompt, image=image, num_frames=124, generator=torch.Generator().manual_seed(42), output=outputs)

encode_video(
    results["videos"][0],
    fps=24,
    output_path="minimax_h3_fl2va.mp4",
    audio=results["audio"][0],
    audio_sample_rate=results["sampling_rate"],
)
