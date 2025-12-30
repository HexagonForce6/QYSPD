import json, yaml, torch, multiprocessing as mp
from multiprocessing import Manager
from tqdm import tqdm
import threading, time

# TODO: tqdm total time est bar should stick to bottom


def flatten_prompts(prompt_path, pair_path, variant_path):
    prompts = json.load(open(prompt_path, encoding="utf-8"))
    pairs   = json.load(open(pair_path, encoding="utf-8"))

    # load variants
    variants_map = {}
    with open(variant_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            variants_map[parts[0]] = parts[1:]

    items = []
    for char, data in prompts.items():
        defs = [d for _, d in pairs.get(char, [])]   # def list
        variants = variants_map.get(char, [])
        for idx, start, is_var, sent in data["occurrences"]:
            items.append({
                "char": char,
                "sentence": sent,
                "is_variant": is_var,
                "variants": variants,
                "defs": defs,
                "idx": idx,        # row index in sentence csv
                "start": start     # starting pos in that row
            })
    return items


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def run_worker(phys_gpu_id, items, cfg, out_path, batch_size=32, counter=None, gpu_stats=None,
               system_prompt="简洁明了地回答问题。"):
    batch_size=cfg["batch_size"]
    import os, json
    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_gpu_id)
    os.environ["VLLM_WORKER_USE_RAY"] = "0"

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    llm = LLM(model=cfg["model_path"], tensor_parallel_size=1, gpu_memory_utilization=0.9)

    sp = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        repetition_penalty=cfg["repetition_penalty"],
        max_tokens=cfg["max_tokens"],
    )

    with open(out_path, "w", encoding="utf-8") as fout:
        pbar = tqdm(total=len(items), desc=f"GPU-{phys_gpu_id}", dynamic_ncols=True)
        total_time, total_occ = 0.0, 0
        
        for batch in chunks(items, batch_size):
            start_time = time.time()
            
            prompts1, metas = [], []

            # Prompt 1: get choice and reasons
            for it in batch:
                char = it["char"]
                variants = it.get("variants", [])
                char_with_variants = char if not it["is_variant"] else f'{char}，又写作"{ " ".join(variants) }"'
                defs = [f"{i+1}. {d}" for i, d in enumerate(it["defs"])]
                def_list = "\n".join(defs)

                p1_text = (
                    f"{char_with_variants}。\n该字有{len(defs)}种释义：\n{def_list}"
                    f'\n在以下句中"{char}"接近以上第几个释义，简要说明原因和答案编号（若以上释义均不准确，回答编号0）：\n{it["sentence"]}'
                )
                messages1 = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": p1_text},
                ]
                chat1 = tokenizer.apply_chat_template(
                    messages1, tokenize=False, add_generation_prompt=True
                )
                prompts1.append(chat1)
                metas.append(it)

            resp1 = llm.generate(prompts1, sampling_params=sp)

            # Prompt 2: extract final choice
            prompts2 = []
            for r1 in resp1:
                r1_text = r1.outputs[0].text.strip().replace("\n","")
                p2_text = f"以下句中，最终选择了答案编号几，仅回答那一个编号数字：\n{r1_text}"
                messages2 = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": p2_text},
                ]
                chat2 = tokenizer.apply_chat_template(
                    messages2, tokenize=False, add_generation_prompt=True
                )
                prompts2.append(chat2)

            resp2 = llm.generate(prompts2, sampling_params=sp)
            
            end_time = time.time()

            elapsed = end_time - start_time
            total_time += elapsed
            total_occ += len(batch)

            avg_time = total_time / total_occ
            pbar.set_postfix_str(f"avg {avg_time:.3f}s/occ")
            pbar.update(len(batch))

            # jsonl output format
            for m, r1, r2 in zip(metas, resp1, resp2):
                rec = {
                    "char": m["char"],
                    "sentence": m["sentence"],
                    "idx": m["idx"],
                    "start": m["start"],
                    "is_variant": m["is_variant"],
                    "variants": m["variants"],
                    "resp1": r1.outputs[0].text,
                    "resp2": r2.outputs[0].text,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
            if counter is not None:
                with counter.get_lock():
                    counter.value += len(batch)
        
        pbar.close()
        if gpu_stats is not None:
            gpu_stats[str(phys_gpu_id)] = {
                "gpu_id": phys_gpu_id,
                "completed_occurrences": total_occ,
                "batches": (total_occ + batch_size - 1) // batch_size,
                "total_time_sec": total_time,
                "avg_sec_per_occ": (total_time / total_occ) if total_occ else 0.0
            }
        

def main(cfg_path="config.yaml"):
    mp.set_start_method("spawn", force=True)

    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    model_cfg = cfg["model"]

    items = flatten_prompts(cfg["prompts"]["prompt_path"],
                            cfg["prompts"]["pair_path"],
                            cfg["prompts"]["variant_path"])

    G = torch.cuda.device_count()
    assert G >= 1, "No GPUs visible"
    chunk = (len(items) + G - 1) // G

    # shared counter for all GPUs
    counter = mp.Value('i', 0)
    done = threading.Event()

    total = len(items)
    pbar_total = tqdm(total=total, desc="ALL", dynamic_ncols=True)
    
    # monitor thread to refresh total bar
    def _monitor():
        last = -1
        while not done.is_set():
            n = counter.value
            if n != last:
                pbar_total.n = n
                pbar_total.refresh()
                last = n
            time.sleep(0.1)

    t = threading.Thread(target=_monitor, daemon=True); t.start()
    
    mgr = Manager()
    gpu_stats = mgr.dict() # per-GPU summaries
    t0 = time.time()
    
    procs = []
    for phys_gpu_id in range(G):
        part = items[phys_gpu_id*chunk : (phys_gpu_id+1)*chunk]
        if not part: continue
        p = mp.Process(
            target=run_worker,
            args=(phys_gpu_id, part, model_cfg, f"out_gpu{phys_gpu_id}.jsonl", 32, counter, gpu_stats),
        )
        p.start()
        procs.append(p)

    for p in procs: p.join()
    done.set(); t.join(timeout=1.0)
    
    t1 = time.time()
    sum_time = sum(v["total_time_sec"] for v in gpu_stats.values()) if len(gpu_stats) else 0.0
    done = counter.value
    summary = {
        "ALL": {
            "total_occurrences": total,
            "completed_occurrences": done,
            "wall_time_sec": t1 - t0,
            "avg_sec_per_occ_weighted": (sum_time / done) if done else 0.0
        },
        "per_gpu": gpu_stats.copy()  # { "0": {...}, "1": {...}, ... }
    }
    with open(cfg["prompts"]["progress_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    pbar_total.n = counter.value; pbar_total.refresh(); pbar_total.close()


if __name__ == "__main__":
    main()
