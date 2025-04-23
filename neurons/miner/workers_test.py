import asyncio
import base64
import time
import typing
import pydantic
import urllib.parse
from pydantic import BaseModel, Field
import aiohttp
import bittensor as bt
from typing import Optional
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


async def async_gradio_client(endpoint: str, prompt: str):
    url = f"{endpoint}/api/text_to_3d"
    headers = {"Content-type": "application/json"}
    seed=random.randint(0, 2**32 - 1)
    data = {
        "data": [
          prompt,
          seed,
          6.5, 
          20, 
          4.0, 
          16
        ],
        "fn_index": 3  
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, data=json.dumps(data), headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["data"][1]["video"]["path"]
                else:
                    bt.logging.error(f"Gradio API 错误: [{response.status}] {response.reason}")
                    return None
        except aiohttp.ClientError as e:
            bt.logging.error(f"连接错误: {e}")
            return None
async def _complete_one_task(
    generate_url: list[str], wallet: bt.wallet, metagraph: bt.metagraph, validator_selector: ValidatorSelector
) -> None:
    validator_uid = validator_selector.get_next_validator_to_query()
    if validator_uid is None:
        await asyncio.sleep(2.0)
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
            if cooldown_left >= 500:
                validator_selector.set_cooldown(validator_uid, int(time.time()) + 60)
            else:
                validator_selector.set_cooldown(validator_uid, pull.cooldown_until)
        return

    bt.logging.debug(f"vali_uid :{validator_uid}  获取任务返回 开始多节点生成验证. Prompt: {pull.task.prompt}.")
    validation_results = []
    async def generate_ply(endpoint):
        try:
            client = Client(endpoint)
            random_seed = random.randint(0, 2**32 - 1)
            ply_path = await async_gradio_client(endpoint, pull.task.prompt, random_seed)
            if not ply_path:
              return None, None
            with open(ply_path, 'rb') as file:
                ply_bytes = file.read()
            os.remove(ply_path)
            compresseds = base64.b64encode(pyspz.compress(ply_bytes, workers=-1)).decode(encoding="utf-8")
            vail_url = ["http://127.0.0.1:20000", "http://127.0.0.1:20001"]
            validation_score = await validate(random.choice(vail_url), prompt=pull.task.prompt, results=compresseds, uid=validator_uid)
            return validation_score,compresseds
        except Exception as e:
            bt.logging.error(f"Failed to connect to {endpoint}: {str(e)}")
            return None, None    
    tasks = [generate_ply(endpoint) for endpoint in generate_url]
    validation_results = await asyncio.gather(*tasks)
    best_score = -1.0
    best_results = None
    for validation_score, compresseds in validation_results:
        if validation_score is not None and validation_score > best_score:
            best_score = validation_score
            best_results = compresseds
    async with bt.dendrite(wallet=wallet) as dendrite:
        submit = await _submit_results(wallet, dendrite, metagraph, validator_uid, pull, best_results)
        if submit.feedback is None:
            bt.logging.warning(
                f"vali_uid :{validator_uid}  提交结果出错 to [{metagraph.hotkeys[validator_uid]}]. "
                f"Reason: {submit.dendrite.status_message}."
            )
            validator_selector.set_cooldown(validator_uid, int(time.time()) + FAILED_VALIDATOR_DELAY)
            return 
    _log_feedback(validator_uid, submit)

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
    endpoint: str,prompt: str ,results: str, storage_enabled: bool = False, validation_score_threshold: float = 0.6
) -> Optional[float]:
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
                    "compression": 2,
                    "generate_preview": storage_enabled,
                    "preview_score_threshold": validation_score_threshold - 0.1,
                },
            ) as response:
                if response.status == 200:
                    data_dict = await response.json()
                    result = data_dict
                    print(f"Validation score: {result['score']} | Prompt: {prompt}")
                    return result["score"]
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
        compressed_results = results
        #compressed_results = base64.b64encode(pyspz.compress(results, workers=-1)).decode(encoding="utf-8")
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


def _log_feedback(validator_uid: int, submit: SubmitResults) -> None:
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
        )
    else:
        bt.logging.debug(
            f"平均分数: {feedback.average_fidelity_score}. "
            f"4小时内次数: {feedback.generations_within_the_window}. "
            f"总分数: {feedback.current_miner_reward}."
        )

    
