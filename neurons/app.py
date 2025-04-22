import gradio as gr
import argparse
#import spaces
from gradio_litmodel3d import LitModel3D
import os
import shutil
import random
from openai import OpenAI
import uuid
from datetime import datetime
from diffusers import DiffusionPipeline
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import torch
import time
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils
NUM_INFERENCE_STEPS = 20
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
# Constants
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

# Funciones auxiliares
def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    #shutil.rmtree(user_dir)

def preprocess_image(image: Image.Image) -> Image.Image:
    processed_image = trellis_pipeline.preprocess_image(image)
    return processed_image

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }

def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    return gs, mesh

def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

def generate_flux_image(
    prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    guidance_scale: float,
    
    req: gr.Request,
    num_inference_steps: int = 18,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Image.Image:
    """Generate image using Flux pipeline"""
    start_time = time.time()
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)
    #2. Add PBR material details ONLY if relevant.
    system_content = """
        You are a professional 3D artist responsible for optimizing prompts for generating 3D assets. Rules:
        1. Keep the "wbgmsst" prefix and the "white background, for 3D generation, PBR materials" suffix.
        2. Never change the core object type.
        3. Increase the level of detail in the prompts to make them more suitable for 3D model generation.
        4. Only reply with the optimized prompt.
        5. If the description is unclear, construct the object based on the description.
        """
    client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key="sk-or-v1-*************",)
    completion = client.chat.completions.create(model="deepseek/deepseek-chat-v3-0324",messages=[
        {"role": "system","content": system_content},{"role": "user","content": f"Optimize this prompt for 3D generation: {prompt}"}],temperature=0.5,max_tokens=70)
    print(f"优化前prompt : {prompt}")
    promptrez= prompt
    promptrez = completion.choices[0].message.content
    print(f"优化后prompt : {promptrez}")
    image = flux_pipeline(
        prompt=promptrez,
        guidance_scale=guidance_scale,
        negative_prompt="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions",
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}.png"
    filepath = os.path.join(user_dir, filename)
    image.save(filepath)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"总用时: {total_time:.4f} 秒")
    torch.cuda.empty_cache()
    return image

#@spaces.GPU
def image_to_3d(
    image: Image.Image,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    req: gr.Request,
) -> Tuple[dict, str]:
    start_time = time.time()
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    outputs = trellis_pipeline.run(
        image,
        seed=seed,
        formats=["gaussian","mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    ply_path = os.path.join(user_dir, 'point_cloud.ply')
    gaussian_data = outputs['gaussian'][0]
    with open(ply_path, "wb") as f:
        gaussian_data.save_ply(f)
    state = pack_state(outputs['gaussian'][0],outputs['mesh'][0])
    end_time = time.time()
    total_time = end_time - start_time
    print(f"总用时: {total_time:.4f} 秒")
    torch.cuda.empty_cache()
    return state, ply_path

# Interfaz Gradio
with gr.Blocks() as demo:
    #gr.Markdown("""
    # UTPL - Conversión de Texto a Imagen a objetos 3D usando IA  
    ### Tesis: *"Objetos tridimensionales creados por IA: Innovación en entornos virtuales"*  
    #**Autor:** Carlos Vargas  
    #**Base técnica:** Adaptación de [TRELLIS](https://trellis3d.github.io/) y [FLUX](https://huggingface.co/camenduru/FLUX.1-dev-diffusers) (herramientas de código abierto para generación 3D)
    #**Propósito educativo:** Demostraciones académicas e Investigación en modelado 3D automático  
    #""")
    
    with gr.Row():
        with gr.Column():
            # Flux image generation inputs
            prompt = gr.Text(label="Prompt", placeholder="Enter your game asset description")
            with gr.Accordion("Generation Settings", open=False):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=42, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                with gr.Row():
                    width = gr.Slider(512, 1024, label="Width", value=1024, step=16)
                    height = gr.Slider(512, 1024, label="Height", value=1024, step=16)
                with gr.Row():
                    guidance_scale = gr.Slider(0.0, 10.0, label="Guidance Scale", value=3.5, step=0.1)
                    num_inference_steps = gr.Slider(6, 200, label="num_inference_steps", value=8, step=2)
            # Botones separados
            generate_image_btn = gr.Button("Generar Imagen")
            generate_video_btn = gr.Button("Generar Video", interactive=False)
        with gr.Column():
            generated_image = gr.Image(label="Generated Asset", type="pil")
            ply_viewer = gr.Model3D(label="3D Point Cloud") 
            #video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True)

    # Variables adicionales para la generación 3D
    with gr.Accordion("3D Generation Settings", open=False):
        gr.Markdown("Stage 1: Sparse Structure Generation")
        with gr.Row():
            ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
            ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
        gr.Markdown("Stage 2: Structured Latent Generation")
        with gr.Row():
            slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
            slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
    
    output_buf = gr.State()
    
    # Event handlers
    demo.load(start_session)
    demo.unload(end_session)
    
    # Generar imagen
    generate_image_btn.click(
        generate_flux_image,
        inputs=[prompt, seed, randomize_seed, width, height, guidance_scale,num_inference_steps],
        outputs=[generated_image]
    ).then(
        lambda: gr.Button(interactive=True),  
        outputs=[generate_video_btn],
    )
    
    # Generar video
    generate_video_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        preprocess_image,
        inputs=[generated_image],
        outputs=[generated_image],
    ).then(
        image_to_3d,
        inputs=[
            generated_image, 
            seed, 
            ss_guidance_strength, 
            ss_sampling_steps, 
            slat_guidance_strength, 
            slat_sampling_steps
        ],
        outputs=[output_buf, ply_viewer],
    )
    
# Initialize both pipelines
if __name__ == "__main__":
    from diffusers import FluxTransformer2DModel, FluxPipeline, BitsAndBytesConfig, GGUFQuantizationConfig
    from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF
    parser = argparse.ArgumentParser(description="Gradio app with command-line port argument")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the Gradio app on")
    parser.add_argument("--model", type=str, default="img", help="img or ply")
    args = parser.parse_args()
    port = args.port
    model  = args.model
    # Initialize Flux pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    dtype = torch.bfloat16
    if model == "img":
        file_url = "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf"
        file_url = file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
        single_file_base_model = "camenduru/FLUX.1-dev-diffusers"
        quantization_config_tf = BitsAndBytesConfigTF(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
        text_encoder_2 = T5EncoderModel.from_pretrained(single_file_base_model, subfolder="text_encoder_2", torch_dtype=dtype, config=single_file_base_model, quantization_config=quantization_config_tf, token=huggingface_token)
    
        if ".gguf" in file_url:
            transformer = FluxTransformer2DModel.from_single_file(file_url, subfolder="transformer", quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype, config=single_file_base_model)
        else:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, token=huggingface_token)
            transformer = FluxTransformer2DModel.from_single_file(file_url, subfolder="transformer", torch_dtype=dtype, config=single_file_base_model, quantization_config=quantization_config, token=huggingface_token)
    
        flux_pipeline = FluxPipeline.from_pretrained(single_file_base_model, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=dtype, token=huggingface_token)
        flux_pipeline.to("cuda")
    else:
        # Initialize Trellis pipeline
        trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained("cavargas10/TRELLIS")
        trellis_pipeline.cuda()
    
    
    demo.launch(show_error=True,server_name="0.0.0.0",server_port=port)
