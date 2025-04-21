from gradio_client import Client, handle_file
import urllib.parse
import aiohttp
import pydantic
from pydantic import BaseModel, Field
import asyncio
import base64

def mp4_to_bytes_open(file_path):
  try:
    with open(file_path, 'rb') as f:
      video_bytes = f.read()
    return video_bytes
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    return None
  except Exception as e:
    print(f"Error reading file: {e}")
    return None



class ValidationResponse(BaseModel):
    score: float = Field(default=0.0, description="Validation score, from 0.0 to 1.0")
    iqa: float = Field(default=0.0, description="Aesthetic Predictor (quality) score")
    clip: float = Field(default=0.0, description="Clip similarity score")
    ssim: float = Field(default=0.0, description="Structure similarity score")
    lpips: float = Field(default=0.0, description="Perceptive similarity score")
    preview: str | None = Field(default=None, description="Optional. Preview image, base64 encoded PNG")



async def validate(
    endpoint: str,prompt: str ,results: str, storage_enabled: bool = False, validation_score_threshold: float = 0.6
) -> ValidationResponse | None:
    prompt = prompt  # type: ignore[union-attr]
    data = results
    validate_url = urllib.parse.urljoin(endpoint, "/validate_txt_to_3d_ply/")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                validate_url,
                json={
                    "prompt": prompt,
                    "data": data,
                    "compression": 0,
                    "generate_preview": storage_enabled,
                    "preview_score_threshold": validation_score_threshold - 0.1,
                },
            ) as response:
                if response.status == 200:
                    data_dict = await response.json()
                    results = ValidationResponse(**data_dict)
                    print(f"Validation score: {results.score:.2f} | Prompt: {prompt}")
                    return results
                else:
                    print(f"Validation failed: [{response.status}] {response.reason}")
        except aiohttp.ClientConnectorError:
            print(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {endpoint}.")
        except TimeoutError:
            print(f"The request to the endpoint timed out: {endpoint}")
        except aiohttp.ClientError as e:
            print(f"An unexpected client error occurred: {e} ({endpoint})")
        except pydantic.ValidationError as e:
            print(f"Incompatible validation response format: {e} ({endpoint})")
        except Exception as e:
            print(f"An unexpected error occurred: {e} ({endpoint})")

    return None

async def main():
    client = Client("http://86.***.***.***:44549/")
    result = client.predict(
        prompt="glass table with orchid in center",
        seed=42,
        randomize_seed=True,
        width=512,
        height=512,
        guidance_scale=9.0,
        num_inference_steps=8,
        api_name="/generate_flux_image"
    )
    print(result)
    result = client.predict(
        image=handle_file(result),
        seed=42,
        ss_guidance_strength=8.5,
        ss_sampling_steps=12,
        slat_guidance_strength=3.5,
        slat_sampling_steps=12,
        api_name="/image_to_3d"
    )
    print(result)
    test = mp4_to_bytes_open(result)
    print (len(test))
    compressed_results = base64.b64encode(test).decode(encoding="utf-8")
    validation_res = await validate("http://***.***.216.***:21002", prompt="glass table with orchid in center",results=compressed_results)

    print(validation_res)

if __name__ == "__main__":
    asyncio.run(main())
