import os
import torch
from diffusers import ModularPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video

# Official group offload APIs keep the original BF16 weights on this 64 GiB device.
pipe = ModularPipeline.from_pretrained(os.environ["MODEL_PATH"])
pipe.load_components(workflow="fl2va", dtype=torch.bfloat16, pretrained_model_name_or_path=os.environ["MODEL_PATH"], local_files_only=True)

pipe.transformer.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=False)
pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
pipe.vae.to("cuda")
pipe.audio_vae.to("cuda")

prompt = "A dog falls asleep"
# `output=` returns exactly the named outputs instead of the whole pipeline state.
outputs = ["videos", "audio", "sampling_rate"]

# First frame (and optionally last frame) to video + audio. The canvas follows the first keyframe.
image = load_image(
    os.environ["FIRST_FRAME"]
)
results = pipe(prompt=prompt, image=image, num_frames=124, height=384, width=576, num_inference_steps=10, generator=torch.Generator().manual_seed(42), output=outputs)

encode_video(
    results["videos"][0],
    fps=24,
    output_path=os.environ["OUTPUT_VIDEO"],
    audio=results["audio"][0],
    audio_sample_rate=results["sampling_rate"],
)
