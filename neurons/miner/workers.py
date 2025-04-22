import asyncio
import base64
import time
import typing
import pydantic
import urllib.parse
from pydantic import BaseModel, Field
from gradio_client import Client, handle_file
import aiohttp
import bittensor as bt
import pyspz
from openai import OpenAI
import random
import os
from aiohttp import ClientTimeout
from aiohttp.helpers import sentinel
from common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION
from common.protocol import PullTask, SubmitResults

from miner import ValidatorSelector
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1" 
class ValidationResponse(BaseModel):
    score: float = Field(default=0.0, description="Validation score, from 0.0 to 1.0")
    iqa: float = Field(default=0.0, description="Aesthetic Predictor (quality) score")
    clip: float = Field(default=0.0, description="Clip similarity score")
    ssim: float = Field(default=0.0, description="Structure similarity score")
    lpips: float = Field(default=0.0, description="Perceptive similarity score")
    preview: str | None = Field(default=None, description="Optional. Preview image, base64 encoded PNG")
NETWORK_DELAY_TIME_BUFFER = 60
FAILED_VALIDATOR_DELAY = 301

#/tmp/gradio


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


async def worker_routine(
    endpoint: list[str], wallet: bt.wallet, metagraph: bt.metagraph, validator_selector: ValidatorSelector
) -> None:
    #bt.logging.info(f"Worker ({endpoint}) started")
    while True:
        await _complete_one_task(endpoint, wallet, metagraph, validator_selector)


def _call_gradio_client(endpoint: str, prompt: str, seed: int,client: Client) -> str:
    #client = Client(endpoint)
    try:
        return client.predict(
            prompt=prompt,
            seed=seed,
            randomize_seed=True,
            width=512,
            height=512,
            guidance_scale=8.0,
            num_inference_steps=12,
            api_name="/generate_flux_image"
        )
    finally:
        client.close()

def _call_gradio_client_image_to_3d(endpoint: str, image_path: str, seed: int,client: Client) -> str:
    #client = Client(endpoint)
    try:
        return client.predict(
            image=handle_file(image_path),
            seed=seed,
            ss_guidance_strength=8.5,
            ss_sampling_steps=12,
            slat_guidance_strength=3.5,
            slat_sampling_steps=12,
            api_name="/image_to_3d"
        )
    finally:
        client.close()

async def _complete_one_task(
    generate_url: list[str], wallet: bt.wallet, metagraph: bt.metagraph, validator_selector: ValidatorSelector
) -> None:
    validator_uid = validator_selector.get_next_validator_to_query()
    if validator_uid is None:
        await asyncio.sleep(10.0)
        return
    # Setting cooldown to prevent selecting the same validator for concurrent task.
    #validator_selector.set_cooldown(validator_uid, int(time.time()) + 300)

    async with bt.dendrite(wallet=wallet) as dendrite:
        pull = await _pull_task(dendrite, metagraph, validator_uid)
        #bt.logging.debug(f"validator_uid :{validator_uid}   pull received : {pull} ")
        if pull.dendrite.status_code != 200:
            bt.logging.warning(f"validator_uid :{validator_uid} Failed to get task. Reason: {pull.dendrite.status_message}.")
            if pull.cooldown_until == 0:
                validator_selector.set_cooldown(validator_uid, int(time.time()) + 20)
            else:
                validator_selector.set_cooldown(validator_uid, pull.cooldown_until)
            return

    if pull.task is None:
        if pull.cooldown_until == 0:
            bt.logging.warning(f"vali_uid :{validator_uid}  Failed to get task. Reason: Unknown.")
            validator_selector.set_cooldown(validator_uid, int(time.time()) + 20)
        else:
            cooldown_left = max(0, int(pull.cooldown_until - time.time()))
            bt.logging.debug(
                f"vali_uid :{validator_uid}  Miner 在冷却期 : {cooldown_left} sec. "
                f"总冷却次数: {pull.cooldown_violations}"
            )
            validator_selector.set_cooldown(validator_uid, pull.cooldown_until)
        return

    bt.logging.debug(f"vali_uid :{validator_uid}  获取任务返回. Prompt: {pull.task.prompt}.")
    cs = 0
    while True:
        if cs == 4:
            bt.logging.debug(f"vali_uid :{validator_uid} 超过3次低分，跳过 ")
            results = b'' 
            break
        random_seed = random.randint(0, 2**32 - 1)
        endpoints = random.choice(generate_url)
        ply_endpoint = ["http://127.0.0.1:10006","http://127.0.0.1:10007"]
        try:
            
            client = Client(endpoints)
            images = await asyncio.to_thread(
                _call_gradio_client,
                endpoints,
                pull.task.prompt,
                random_seed,
                client
            )
            client = Client(random.choice(ply_endpoint))
            random_seed = random.randint(0, 2**32 - 1)
            vresult = await asyncio.to_thread(
                _call_gradio_client_image_to_3d,
                endpoints,
                images,
                random_seed,
                client
            )
        except Exception as e:
            bt.logging.error(f"Failed to connect to {endpoints}: {str(e)}")
            continue
        os.remove(images)
        results = mp4_to_bytes_open(vresult)
        os.remove(vresult)
        compressed_results = base64.b64encode(results).decode(encoding="utf-8")
        vail_url = ["http://127.0.0.1:20000","http://127.0.0.1:20001"]
        validation_res = await validate(random.choice(vail_url), prompt=pull.task.prompt,results=compressed_results,uid=validator_uid)
        cs = cs + 1
        if validation_res is not None:
            if validator_uid == 49:
                if validation_res.score >= 0.79999:
                    bt.logging.debug(f"vali_uid :{validator_uid} Prompt: {pull.task.prompt} 分数大于0.8 跳出循环提交...")
                    break
            else:

                if validation_res.score >= 0.84999:
                    bt.logging.debug(f"vali_uid :{validator_uid} Prompt: {pull.task.prompt} 分数大于0.85 跳出循环提交...")
                    break

    #bt.logging.debug(f"video received. path: {vresult}. len: {len(results)}")
    
    async with bt.dendrite(wallet=wallet) as dendrite:
        submit = await _submit_results(wallet, dendrite, metagraph, validator_uid, pull, results)
        if submit.feedback is None:
            bt.logging.warning(
                f"vali_uid :{validator_uid}  提交结果出错 to [{metagraph.hotkeys[validator_uid]}]. "
                f"Reason: {submit.dendrite.status_message}."
            )
            validator_selector.set_cooldown(validator_uid, int(time.time()) + FAILED_VALIDATOR_DELAY)
            return 
    _log_feedback(validator_uid, submit,vresult)

    validator_selector.set_cooldown(validator_uid, submit.cooldown_until)


async def _pull_task(dendrite: bt.dendrite, metagraph: bt.metagraph, validator_uid: int) -> PullTask:
    synapse = PullTask()
    response = typing.cast(
        PullTask,
        await dendrite.call(
            target_axon=metagraph.axons[validator_uid], synapse=synapse, deserialize=False, timeout=10.0
        ),
    )
    return response

async def validate(
    endpoint: str,prompt: str ,results: str, storage_enabled: bool = False, validation_score_threshold: float = 0.6,uid = None
) -> ValidationResponse | None:
    prompt = prompt  # type: ignore[union-attr]
    data = results
    validate_url = urllib.parse.urljoin(endpoint, "/validate_txt_to_3d_ply/")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=100)) as session:
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
                    bt.logging.debug(f"vuid : {uid} 本地验证分数: {results.score:.2f} | 提示词: {prompt}")
                    return results
                else:
                    bt.logging.debug(f"vuid : {uid} 本地验证错误: [{response.status}] {response.reason}")
        except aiohttp.ClientConnectorError:
            bt.logging.warning(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {endpoint}.")
        except TimeoutError:
            bt.logging.warning(f"The request to the endpoint timed out: {endpoint}")
        except aiohttp.ClientError as e:
            bt.logging.warning(f"An unexpected client error occurred: {e} ({endpoint})")
        except pydantic.ValidationError as e:
            bt.logging.warning(f"Incompatible validation response format: {e} ({endpoint})")
        except Exception as e:
            bt.logging.warning(f"An unexpected error occurred: {e} ({endpoint})")

    return None
async def _submit_results(
    wallet: bt.wallet,
    dendrite: bt.dendrite,
    metagraph: bt.metagraph,
    validator_uid: int,
    pull: PullTask,
    results: bytes,
) -> SubmitResults:
    submit_time = time.time_ns()
    prompt = pull.task.prompt if pull.task is not None else None
    message = (
        f"{MINER_LICENSE_CONSENT_DECLARATION}"
        f"{submit_time}{prompt}{metagraph.hotkeys[validator_uid]}{wallet.hotkey.ss58_address}"
    )
    signature = base64.b64encode(dendrite.keypair.sign(message)).decode(encoding="utf-8")
    if results:
        compressed_results = base64.b64encode(pyspz.compress(results, workers=-1)).decode(encoding="utf-8")
        #compressed_results = base64.b64encode(results).decode(encoding="utf-8")
    else:
        compressed_results = ""  # Skipping task not to be penalized (same could be done for low quality results)
    synapse = SubmitResults(
        task=pull.task, results=compressed_results, compression=2, submit_time=submit_time, signature=signature
    )
    response = typing.cast(
        SubmitResults,
        await dendrite.call(
            target_axon=metagraph.axons[validator_uid],
            synapse=synapse,
            deserialize=False,
            timeout=300.0,
        ),
    )
    return response


def _log_feedback(validator_uid: int, submit: SubmitResults,vresult : str) -> None:
    feedback = submit.feedback
    if feedback is None:
        return
    score = "failed" if feedback.validation_failed else feedback.task_fidelity_score
    bt.logging.debug(f"收到的反馈来自[{validator_uid}]. Prompt: {submit.task.prompt}. Score: {score}")
    if score==0:
        bt.logging.debug(
        f"平均分数: {feedback.average_fidelity_score}. "
        f"4小时内次数: {feedback.generations_within_the_window}. "
        f"总分数: {feedback.current_miner_reward}."
        f"path: {vresult}."
        )
    else:
        bt.logging.debug(
            f"平均分数: {feedback.average_fidelity_score}. "
            f"4小时内次数: {feedback.generations_within_the_window}. "
            f"总分数: {feedback.current_miner_reward}."
        )

    


async def _generate(generate_url: str, prompt: str, timeout: float | None = None) -> bytes | None:  # noqa: ASYNC109
    bt.logging.debug(f"Generating for prompt: {prompt} with timeout {timeout} seconds")

    client_timeout = ClientTimeout(total=timeout) if timeout is not None else sentinel
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        try:
            async with session.post(generate_url, data={"prompt": prompt}) as response:
                if response.status == 200:
                    results = await response.read()
                    bt.logging.debug(f"Generation completed. Size: {len(results)}")
                    return results
                else:
                    bt.logging.error(f"Generation failed with code: {response.status}")
        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {generate_url}.")
        except TimeoutError:
            bt.logging.error(f"The request to the endpoint timed out: {generate_url}")
        except aiohttp.ClientError as e:
            bt.logging.error(f"An unexpected client error occurred: {e} ({generate_url})")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e} ({generate_url})")
