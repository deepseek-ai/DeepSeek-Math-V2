import json
import orjson
import os

from math_templates import math_templates
from utils import hash_problem_idx, read_data, extract_boxed_answers, extract_solution, extract_self_eval

import itertools
from copy import deepcopy
from glob import glob
import numpy as np
from functools import partial
import math
from tqdm import tqdm
import multiprocessing

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_paths", required=True)
parser.add_argument("--output_dirname", required=True, help="directory to dump results")
parser.add_argument("--proof_pool_dirname", required=True, help="directory to maintain a pool of generated proofs for each evaluated problem")

parser.add_argument("--batch_size", type=int, default=160)

parser.add_argument("--proof_gen_num_processes", type=int, default=40)
parser.add_argument("--proof_gen_temp", type=float, default=1.0)
parser.add_argument("--proof_gen_max_len", type=int, default=128 * 1024)
parser.add_argument("--proof_gen_template", type=str, default="proof_generation")
parser.add_argument("--proof_refine_template", type=str, default="proof_refinement")
parser.add_argument("--n_best_proofs_to_sample", type=int, default=32, help="the number of best proofs to consider for refinements")
parser.add_argument("--n_proofs_to_refine", type=int, default=1, help="the number of proofs used as the input for a refinement")
parser.add_argument("--n_agg_trials", type=int, default=32, help="the number of different combinations of proofs used for refinements")
parser.add_argument("--n_parallel_proof_gen", type=int, default=128)


parser.add_argument("--proof_verification_num_processes", type=int, default=320)
parser.add_argument("--proof_verification_temp", type=float, default=1.0)
parser.add_argument("--proof_verification_max_len", type=int, default=64 * 1024)
parser.add_argument("--proof_verification_template", type=str, default="proof_verification")
parser.add_argument("--n_verification_per_proof", type=int, default=4)


parser.add_argument("--skip_meta_verification", action='store_true')
parser.add_argument("--meta_verification_num_processes", type=int, default=320)
parser.add_argument("--meta_verification_temp", type=float, default=1.0)
parser.add_argument("--meta_verification_max_len", type=int, default=64 * 1024)
parser.add_argument("--meta_verification_template", type=str, default="meta_verification")
parser.add_argument("--n_meta_verification_per_rating", type=int, default=1)


parser.add_argument("--start_round", type=int, default=1)
parser.add_argument("--max_rounds", type=int, default=20)

args, _ = parser.parse_known_args()

input_paths = args.input_paths
output_dirname = args.output_dirname
proof_pool_dirname = args.proof_pool_dirname

proof_gen_url = args.proof_gen_url
proof_rate_url = args.proof_rate_url

args.proof_gen_with_self_eval = args.proof_gen_template in ['proof_generation']

def prepare_proof_verification(path, tar_path):
    print(f"Proof Verification >>>\ninput path = {path}\noutput_path = {tar_path}", flush=True)
    items = read_data(path)
    data = []
    for item in items:
        item['proof_finish_reason'] = item.pop('finish_reason').lower()
        statement = item['question'].strip()
        prover_output = item['output'].strip()
        if item['proof_finish_reason'] == 'stop':
            assert '</think>' in prover_output
            proof = prover_output.split("</think>")[-1].strip()
        else:
            continue
        item['prover_output'] = prover_output

        self_eval = 'null'
        self_eval_score = 0
        if item['proof_finish_reason'] == 'stop' and args.proof_gen_with_self_eval:
            try:
                self_eval = extract_self_eval(proof).strip()
                proof = extract_solution(proof).strip()
                try:
                    self_eval_score = float([s.strip() for s in extract_boxed_answers(self_eval) if s.strip()][-1])
                except Exception:
                    self_eval_score = 0
            except Exception:
                continue

            item['self_eval'] = self_eval
            item['self_eval_score'] = self_eval_score

        item['proof'] = proof

        question = math_templates[args.proof_verification_template].format(
            statement=statement.strip(),
            proof=proof.strip()
        )
        item.update({
            'messages': [
                {'role': 'user', 'content': question},
            ]
        })
        for key in ['finished', 'finish_reason', 'input', 'output']:
            if key in item:
                item.pop(key)
        data.append(item)
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    with open(tar_path, "w") as file:
        for item in data:
            print(json.dumps(item), file=file, flush=True)
    return len(data)

def prepare_meta_verification(path, tar_path, drop_thought=True):
    print(f"Meta Verification >>>\ninput path = {path}\noutput_path = {tar_path}", flush=True)
    items = read_data(path)
    data = []
    for item in items:
        problem = item['question'].strip()
        if item['finish_reason'] == 'stop' and '</think>' in item['output']:
            rating = item['output'].strip()
            if drop_thought and '</think>' in rating:
                rating = rating.split("</think>")[-1].strip()
            scores = [s.strip() for s in extract_boxed_answers(rating) if s.strip()]
            try:
                score = float(scores[-1])
            except Exception:
                continue
            if score > 0.75:
                continue
            inp = math_templates[args.meta_verification_template].format(
                statement=problem.strip(),
                proof=item['proof'].strip(),
                rating=rating.strip()
            )
            item.update({
                'messages': [
                    {'role': 'user', 'content': inp}
                ],
                'rating': rating
            })
            for key in ['finished', 'finish_reason', 'input', 'output']:
                if key in item:
                    item.pop(key)
            data.append(item)
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    with open(tar_path, "w") as file:
        for item in data:
            print(json.dumps(item), file=file, flush=True)
    return len(data)

def _split_jobs(jobs, nsplit):
    if len(jobs) < nsplit:
        return [jobs]
    res = []
    sz = math.ceil(len(jobs) / nsplit)
    for i in range(0, len(jobs), sz):
        res.append(jobs[i: i + sz])
    return res

def _prepare_proof_agg_tasks(tasks, round_idx=None, proof_pool_dirname=None, use_old_proofs_for_refinement=False, num_trials=16, n_best_proofs_to_sample=6, n_proofs_to_refine=4, max_rating_per_score=4):
    data = []
    trials = []
    print(f"tasks = {len(tasks[0])}", flush=True)
    for (item, proof2ratings, proof2self_eval, proof2dep_proof_ids) in tasks:
        source_name = item.get('source_name', 'temp_source_name')
        if 'problem_idx' in item:
            problem_idx = str(item['problem_idx'])
        else:
            problem_idx = hash_problem_idx(item['question'].strip())
        problem = item['question']
        old_proof_pool = []
        proof_dedup = set()
        proof_id_dedup = set()
        proof_pool_path = f"{proof_pool_dirname}/{source_name}/{problem_idx}.jsonl"
        if os.path.exists(proof_pool_path):
            with open(proof_pool_path, "r") as file:
                for line in file:
                    record = json.loads(line)
                    assert (problem_idx, record['proof']) not in proof_dedup
                    assert record.get('proof_id', 'null') not in proof_id_dedup
                    old_proof_pool.append(
                        (record['proof'], record['meanscore'], record['score2ratings'], record['self_eval'], record.get('proof_id', 'null'))
                    )
                    proof_dedup.add((problem_idx, record['proof']))
                    proof_id_dedup.add(record.get('proof_id', 'null'))
        if proof_id_dedup:
            nxt_proof_id = max(proof_id_dedup) + 1
        else:
            nxt_proof_id = 1
        proof_meanscore_ratings_tuples = []
        for proof, ratings in proof2ratings.items():
            if (problem_idx, proof) in proof_dedup:
                continue
            meanscore = float(np.mean([rating['score'] for rating in ratings]))
            score2ratings = {}
            for rating in ratings:
                score = rating['score']
                if score not in score2ratings:
                    score2ratings[score] = []
                score2ratings[score].append(rating)
            proof_dedup.add((problem_idx, proof))
            proof_id = nxt_proof_id
            nxt_proof_id += 1
            record = (proof, meanscore, score2ratings, proof2self_eval[proof], proof_id, proof2dep_proof_ids[proof])
            proof_meanscore_ratings_tuples.append(record)
        os.makedirs(os.path.dirname(proof_pool_path), exist_ok=True)
        with open(proof_pool_path, "a") as file:
            for record in proof_meanscore_ratings_tuples:
                record = dict(zip(['proof', 'meanscore', 'score2ratings', 'self_eval', 'proof_id', 'dep_proof_ids'], record))
                record['round_idx'] = round_idx
                for _id in record['dep_proof_ids']:
                    assert _id in proof_id_dedup or float(_id) < 0, f"{_id} {len(proof_id_dedup)} {proof_pool_path}"
                print(json.dumps(record), file=file, flush=True)
        if use_old_proofs_for_refinement:
            proof_meanscore_ratings_tuples += old_proof_pool

        if any(record[1] > 0.99999 for record in proof_meanscore_ratings_tuples):
            continue

        for _ in range(10):
            np.random.shuffle(proof_meanscore_ratings_tuples)
        proof_meanscore_ratings_tuples = sorted(proof_meanscore_ratings_tuples, key=lambda x: (x[1], x[3]['self_eval_score']), reverse=True)[:n_best_proofs_to_sample]
        combinations = [list(range(min(n_proofs_to_refine, len(proof_meanscore_ratings_tuples))))] + \
            list(itertools.combinations(list(range(min(n_best_proofs_to_sample, len(proof_meanscore_ratings_tuples)))), min(n_proofs_to_refine, len(proof_meanscore_ratings_tuples))))
        if not proof_meanscore_ratings_tuples:
            combinations = []
        dedup = set()
        for i, indices in enumerate(combinations):
            if len(dedup) == num_trials:
                break
            indices = list(indices)
            if i > 0:
                np.random.shuffle(indices)
            for num_proofs_to_include in range(n_proofs_to_refine, 0, -1):
                if tuple(sorted(indices[:num_proofs_to_include])) in dedup:
                    break
                summary = []
                dep_proof_ids = []
                for idx in indices[:num_proofs_to_include]:
                    proof, meanscore, score2ratings, self_eval, proof_id = proof_meanscore_ratings_tuples[idx][:5]
                    dep_proof_ids.append(proof_id)
                    score2ratings = {float(key): val for key, val in score2ratings.items()}
                    scores = sorted(list(score2ratings.keys()))
                    if len(scores) == 1:
                        max_rating = 8
                    else:
                        max_rating = max_rating_per_score
                    ratings = []
                    for score in scores:
                        assert isinstance(score, float), score
                        np.random.shuffle(score2ratings[score])
                        for rating in score2ratings[score][:max_rating]:
                            rating = rating['rating']
                            ratings.append(f"=== Evaluation {len(ratings)} of Solution {len(summary)} ===\n{rating}")
                            if len(ratings) == 8:
                                break
                    ratings = "\n\n".join(ratings)
                    summary.append(f"--- Solution {len(summary)} ---\n{proof}\n\n{ratings}")
                summary = "\n\n\n".join(summary)
                msg = [
                    {
                        'role': 'user',
                        'content': math_templates[args.proof_refine_template].format(
                            instruction=math_templates[args.proof_gen_template].format(question=problem.strip()).strip(),
                            proofs_to_refine=summary.strip()
                        )
                    }
                ]
                dedup.add(tuple(sorted(indices[:num_proofs_to_include])))
                sample = deepcopy(item)
                sample.update({
                    'messages': msg,
                    'dep_proof_ids': dep_proof_ids
                })
                data.append(sample)
                break

        trials.append(len(dedup))
    return data, trials

def prepare_proof_refinement(
    path, meta_verification_path, tar_path, round_idx,
    proof_pool_dirname=None, use_old_proofs_for_refinement=False, num_trials=16, n_best_proofs_to_sample=6, n_proofs_to_refine=4, max_rating_per_score=4, drop_thought=True
):
    print(f"Proof refinement >>>\ninput path = {path}\noutput_path = {tar_path}\nproof_pool_dirname = {proof_pool_dirname}", flush=True)

    problem2item = {}
    problem2proof2ratings = {}
    problem2proof2self_eval = {}
    problem2proof2dep_proof_ids = {}

    rating2quality = {}
    if os.path.exists(meta_verification_path):
        with open(meta_verification_path, "r") as file:
            for line in tqdm(file, desc='reading meta verification outputs'):
                item = orjson.loads(line)
                rating = item['rating'].strip()
                if item['finish_reason'] == 'stop' and '</think>' in item['output']:
                    quality = item['output'].strip()
                    if drop_thought and '</think>' in quality:
                        quality = quality.split("</think>")[-1].strip()
                    scores = [s.strip() for s in extract_boxed_answers(quality) if s.strip()]
                    try:
                        score = float(scores[-1])
                    except Exception:
                        continue
                    if rating not in rating2quality:
                        rating2quality[rating] = []
                    rating2quality[rating].append({
                        'quality': quality,
                        'score': score
                    })

    with open(path, "r") as file:
        for line in tqdm(file, desc='reading proof verification outputs'):
            item = orjson.loads(line)
            problem = item['question'].strip()
            prover_output = item['proof'].strip()
            if drop_thought and '</think>' in prover_output:
                prover_output = prover_output.split("</think>")[-1].strip()
            if item['finish_reason'] == 'stop' and '</think>' in item['output']:
                rating = item['output'].strip()
                if drop_thought and '</think>' in rating:
                    rating = rating.split("</think>")[-1].strip()
                scores = [s.strip() for s in extract_boxed_answers(rating) if s.strip()]
                try:
                    score = float(scores[-1])
                except Exception:
                    continue
                if problem not in problem2proof2ratings:
                    problem2proof2ratings[problem] = {}
                    problem2proof2self_eval[problem] = {}
                    problem2proof2dep_proof_ids[problem] = {}
                    problem2item[problem] = {key: val for key, val in item.items() if key not in ['messages', 'output', 'input', 'finish_reason', 'meta', 'finished']}
                if prover_output not in problem2proof2ratings[problem]:
                    problem2proof2ratings[problem][prover_output] = []
                    problem2proof2self_eval[problem][prover_output] = {
                        'self_eval': item.get('self_eval', 'null'),
                        'self_eval_score': item.get('self_eval_score', 0)
                    }
                    problem2proof2dep_proof_ids[problem][prover_output] = item.get('dep_proof_ids', [])
                problem2proof2ratings[problem][prover_output].append({
                    'rating': rating,
                    'quality': rating2quality.get(rating, []),
                    'score': score
                })

    print(f"Num statements loaded = {len(problem2proof2ratings)}", flush=True)

    problem_idx_dedup = set()
    tasks = []
    for problem, proof2ratings in problem2proof2ratings.items():
        item = problem2item[problem]
        if 'problem_idx' in item:
            problem_idx = str(item['problem_idx'])
        else:
            problem_idx = hash_problem_idx(item['question'].strip())
        assert problem_idx not in problem_idx_dedup, problem_idx
        problem_idx_dedup.add(problem_idx)
        proof2self_eval = problem2proof2self_eval[problem]
        proof2dep_proof_ids = problem2proof2dep_proof_ids[problem]
        tasks.append(
            (item, proof2ratings, proof2self_eval, proof2dep_proof_ids)
        )

    _args = dict(
        round_idx=round_idx,
        proof_pool_dirname=proof_pool_dirname,
        use_old_proofs_for_refinement=use_old_proofs_for_refinement,
        num_trials=num_trials,
        n_best_proofs_to_sample=n_best_proofs_to_sample,
        n_proofs_to_refine=n_proofs_to_refine,
        max_rating_per_score=max_rating_per_score,
    )
    data = []
    trials = []
    cpu_count = multiprocessing.cpu_count()
    print(f"multiprocessing: {cpu_count} workers", flush=True)
    pool = multiprocessing.Pool(cpu_count)
    for (_data, _trials) in tqdm(pool.imap(partial(_prepare_proof_agg_tasks, **_args), _split_jobs(tasks, 50))):
        data.extend(_data)
        trials.extend(_trials)

    print(f"Avg trials per statement = {np.mean(trials)}", flush=True)

    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    with open(tar_path, "w") as file:
        for item in data:
            print(json.dumps(item), file=file, flush=True)
    return len(data)

if __name__ == '__main__':
    for R in range(args.start_round, args.max_rounds + 2):
        proof_gen_input_path = f"{output_dirname}/proof_gen_R{R}/input.jsonl"
        proof_gen_output_path = f"{output_dirname}/proof_gen_R{R}/output.jsonl"
        if not os.path.exists(proof_gen_input_path):
            if R == 1:
                raw_data = []
                for input_path in input_paths.split(","):
                    source_name = input_path.split("/")[-1].split(".")[0]
                    if input_path.endswith(".json"):
                        _local_data = json.load(open(input_path, "r"))
                    else:
                        _local_data = []
                        with open(input_path, "r") as file:
                            for line in file:
                                _local_data.append(json.loads(line))
                    for item in _local_data:
                        item['source_name'] = source_name
                    raw_data.extend(_local_data)
                data = []
                for item in raw_data:
                    question = math_templates[args.proof_gen_template].format(question=item['question'].strip())
                    item.update({
                        'messages': [
                            {'role': 'user', 'content': question}
                        ]
                    })
                    data.append(item)
                os.makedirs(os.path.dirname(proof_gen_input_path), exist_ok=True)
                with open(proof_gen_input_path, "w") as file:
                    for item in data:
                        print(json.dumps(item), file=file, flush=True)
            else:
                previous_proof_verification_output_path = f"{output_dirname}/proof_verification_R{R - 1}/output.jsonl"
                previous_meta_verification_output_path = f"{output_dirname}/meta_verification_R{R - 1}/output.jsonl"
                prepare_proof_refinement(
                    path=previous_proof_verification_output_path,
                    meta_verification_path=previous_meta_verification_output_path,
                    tar_path=proof_gen_input_path,
                    round_idx=R - 1,
                    proof_pool_dirname=proof_pool_dirname,
                    use_old_proofs_for_refinement=True,
                    num_trials=args.n_agg_trials,
                    n_best_proofs_to_sample=args.n_best_proofs_to_sample,
                    n_proofs_to_refine=args.n_proofs_to_refine,
                    max_rating_per_score=4,
                    drop_thought=True
                )
                if R == args.max_rounds + 1:
                    break

        n_sample = args.n_parallel_proof_gen if R == 1 else args.n_parallel_proof_gen // args.n_agg_trials
        proof_gen_cmd = f"""
    python {args.infer_script}.py \
    --input_data_path {proof_gen_input_path} \
    --output_data_path {proof_gen_output_path} \
    --api_url {proof_gen_url} \
    --batch_size {args.batch_size} \
    --num_processes {args.proof_gen_num_processes} \
    --temperature {args.proof_gen_temp} \
    --top_p 0.95 \
    --max_tokens {args.proof_gen_max_len} \
    --n {n_sample}
    """.strip()
        print(proof_gen_cmd, flush=True)
        os.system(proof_gen_cmd)









        proof_verification_input_path = f"{output_dirname}/proof_verification_R{R}/input.jsonl"
        proof_verification_output_path = f"{output_dirname}/proof_verification_R{R}/output.jsonl"
        if not os.path.exists(proof_verification_input_path):
            prepare_proof_verification(
                path=proof_gen_output_path,
                tar_path=proof_verification_input_path
            )

        proof_verification_cmd = f"""
    python generate.py \
        --input_data_path {proof_verification_input_path} \
        --output_data_path {proof_verification_output_path} \
        --batch_size {args.batch_size} \
        --num_processes {args.proof_verification_num_processes} \
        --temperature {args.proof_verification_temp} \
        --top_p 0.95 \
        --max_tokens {args.proof_verification_max_len} \
        --n {args.n_verification_per_proof}
    """.strip()
        print(proof_verification_cmd, flush=True)
        os.system(proof_verification_cmd)









        meta_verification_input_path = f"{output_dirname}/meta_verification_R{R}/input.jsonl"
        meta_verification_output_path = f"{output_dirname}/meta_verification_R{R}/output.jsonl"
        if not args.skip_meta_verification:
            if not os.path.exists(meta_verification_input_path):
                prepare_meta_verification(
                    path=proof_verification_output_path,
                    tar_path=meta_verification_input_path
                )

            meta_verification_cmd = f"""
    python generate.py \
        --input_data_path {meta_verification_input_path} \
        --output_data_path {meta_verification_output_path} \
        --batch_size {args.batch_size} \
        --num_processes {args.meta_verification_num_processes} \
        --temperature {args.meta_verification_temp} \
        --top_p 0.95 \
        --max_tokens {args.meta_verification_max_len} \
        --n {args.n_meta_verification_per_rating}
    """.strip()
            print(meta_verification_cmd, flush=True)
            os.system(meta_verification_cmd)
