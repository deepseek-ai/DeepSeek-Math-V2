import os
import json
import pickle
import math
import argparse
import asyncio
import aiohttp

from tqdm import tqdm
from multiprocessing import Queue, Process
from time import time, sleep

from openai import AsyncOpenAI

class APIModel:
    def __init__(self, api_url=None):
        base_url = api_url or os.environ.get("OPENAI_BASE_URL", "yyy")
        api_key = os.environ.get("OPENAI_API_KEY", "xxx")
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=300000,
            base_url=base_url
        )

    async def generate_one(self, prompt, sampling_params):
        res = await self.client.chat.completions.create(
            messages=prompt,
            stream=False,
            **sampling_params
        )
        reasoning_content = res.choices[0].message.reasoning_content.strip()
        content = res.choices[0].message.content.strip()
        output_string = f"<think>\n{reasoning_content}"
        if content:
            output_string = reasoning_content + f"\n</think>\n{content}"
        finish_reason = res.choices[0].finish_reason
        return output_string, finish_reason

    async def generate_all(self, data):
        tasks = [self.generate_one(task['prompt'], task['sampling_params']) for i, task in enumerate(data)]
        results = await asyncio.gather(*tasks)
        return results

    def generate(self, input_data, sampling_params):
        data = []
        for item in input_data:
            if "messages" not in item:
                messages = [{
                    "role": "user",
                    "content": item["prompt"],
                }]
            else:
                messages = item['messages']
            data.append({
                'prompt': messages,
                'sampling_params': sampling_params
            })

        outputs = asyncio.run(self.generate_all(data))
        output_data = []
        assert len(input_data) == len(outputs)
        for item, (output_string, finish_reason) in zip(input_data, outputs):
            output_data.append({
                **item,
                "output": output_string,
                "finish_reason": finish_reason.lower(),
            })
        return output_data

    def mp_generate(self, input_queue: Queue, output_queue: Queue, sampling_params):
        while True:
            batch_idx, input_data = input_queue.get()
            if input_data is None:
                output_queue.put((batch_idx, None))
                break
            output_data = self.generate(input_data, sampling_params)
            output_queue.put((batch_idx, output_data))


def mp_generate_loop(input_queue, output_queue, sampling_params, api_url=None):
    api_model = APIModel(api_url=api_url)
    sleep(5)
    api_model.mp_generate(input_queue, output_queue, sampling_params)

def get_parser():
    parser = argparse.ArgumentParser(description="DeepSeekMath-V2 batch inference generator")
    parser.add_argument("--input_data_path", required=True, help="path to input jsonl")
    parser.add_argument("--output_data_path", required=True, help="path to output jsonl")
    parser.add_argument("--api_url", default=None, type=str, help="OpenAI-compatible API base URL")
    parser.add_argument("--num_processes", default=16, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--top_p", required=True, type=float)
    parser.add_argument("--max_tokens", required=True, type=int)
    parser.add_argument("--n", required=True, type=int)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args, _ = parser.parse_known_args()
    input_data_path, output_data_path = args.input_data_path, args.output_data_path
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

    num_processes = args.num_processes
    batch_size = args.batch_size
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens
    n = args.n

    meta_data_path = f"{output_data_path}.meta"
    if not os.path.exists(meta_data_path):
        meta_data = {"n": n, "batch_size": batch_size, "complete_batches": []}
        with open(meta_data_path, "wb") as f:
            pickle.dump(meta_data, f)
    with open(meta_data_path, "rb") as f:
        meta_data = pickle.load(f)
    meta_data["complete_batches"] = set(meta_data["complete_batches"])

    assert n == meta_data["n"] and batch_size == meta_data["batch_size"], \
        f"params n or batch_size are different from previous running setting({n}, {batch_size}) != ({meta_data['n']}, {meta_data['batch_size']}), you need to delete {output_data_path} & {meta_data_path} to clear existing results"

    sampling_params = dict(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_total_tokens=max_tokens
    )

    input_queue, output_queue = Queue(), Queue()
    fr = open(input_data_path, "r", encoding="utf-8")
    fw = open(output_data_path, "a+", encoding="utf-8")

    processes = []
    
    for i in range(num_processes):
        process = Process(target=mp_generate_loop, args=(input_queue, output_queue, sampling_params, getattr(args, 'api_url', None)))
        process.start()
        processes.append(process)

    submit_batch = []
    num_input = 0
    num_skip = 0
    batch_idx = 0

    for line in tqdm(fr, desc="Waiting Input"):
        item = json.loads(line)
        for i in range(n):
            submit_batch.append(item)
            if len(submit_batch) >= batch_size:
                if batch_idx not in meta_data["complete_batches"]:
                    num_input += batch_size
                    input_queue.put((batch_idx, submit_batch))
                else:
                    num_skip += batch_size
                batch_idx += 1
                submit_batch = []
    if len(submit_batch) > 0:
        if batch_idx not in meta_data["complete_batches"]:
            input_queue.put((batch_idx, submit_batch))
            num_input += len(submit_batch)
        else:
            num_skip += len(submit_batch)
    print(f"Total Input Samples: {num_input} (Skip {num_skip} Samples)")
    fr.close()

    for i in range(num_processes):
        input_queue.put((None, None))

    remain_processes = num_processes
    num_output = 0
    with tqdm(desc="Waiting Output", total=num_input) as pbar:
        while remain_processes > 0:
            batch_idx, output_data = output_queue.get()
            if output_data is None:
                remain_processes -= 1
                continue
            for item in output_data:
                print(json.dumps(item, ensure_ascii=False), file=fw, flush=True)
                num_output += 1
                pbar.update(1)
            meta_data["complete_batches"].add(batch_idx)
            with open(meta_data_path, "wb") as f:
                pickle.dump(meta_data, f)
            fw.flush()
    print(f"Total Output Samples: {num_output}")
    fw.close()
    [process.join() for process in processes]
