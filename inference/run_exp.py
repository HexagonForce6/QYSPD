import json
import multiprocessing as mp
import os
import re
import threading
import time
from copy import deepcopy
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path

import torch
import yaml
from tqdm import tqdm



SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_WRITE_INTERVAL_SEC = 1.0
DEFAULT_SYSTEM_PROMPT = "简洁明了地回答问题。"


def resolve_config_path(cfg_path):
    raw_path = Path(cfg_path).expanduser()
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (SCRIPT_DIR / raw_path).resolve()


def resolve_input_path(path_str, *roots):
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        return raw_path

    seen = set()
    candidates = []
    for root in roots:
        if root is None:
            continue
        candidate = (Path(root) / raw_path).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else raw_path.resolve()


def resolve_model_path(path_str, *roots):
    raw_text = str(path_str).strip()
    raw_path = Path(raw_text).expanduser()

    # Plain Hugging Face repo ids like "org/model" should pass through unchanged.
    is_repo_id_like = (
        not raw_path.is_absolute()
        and not raw_text.startswith(".")
        and "/" in raw_text
        and raw_text.count("/") == 1
        and "\\" not in raw_text
    )
    if is_repo_id_like:
        return raw_text

    resolved = resolve_input_path(raw_text, *roots)
    if resolved.exists():
        return str(resolved)

    raise FileNotFoundError(
        f"Model path does not exist: {raw_text!r} -> {resolved}"
    )


def resolve_output_path(path_str):
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        return raw_path
    return (Path.cwd() / raw_path).resolve()


def slugify(value):
    text = str(value).strip()
    text = re.sub(r"[\\/]+", "-", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", text)
    return text.strip("-_.") or "unknown"


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def atomic_write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    os.replace(tmp_path, path)


def make_run_dir(model_cfg, prompt_cfg, progress_alias_path):
    model_tag = slugify(Path(model_cfg["model_path"]).name or model_cfg["model_path"])
    prompt_tag = slugify(Path(prompt_cfg["prompt_path"]).stem)
    run_name = "__".join([
        prompt_tag,
        model_tag,
        f"temp-{slugify(model_cfg['temperature'])}",
        f"top_p-{slugify(model_cfg['top_p'])}",
        f"rep-{slugify(model_cfg['repetition_penalty'])}",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    ])

    run_dir = progress_alias_path.parent / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = progress_alias_path.parent / f"{run_name}-{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def flatten_prompts(prompt_path, pair_path, variant_path):
    with open(prompt_path, encoding="utf-8") as f:
        prompts = json.load(f)
    with open(pair_path, encoding="utf-8") as f:
        pairs = json.load(f)

    variants_map = {}
    with open(variant_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            variants_map[parts[0]] = parts[1:]

    items = []
    for char, data in prompts.items():
        defs = [d for _, d in pairs.get(char, [])]
        variants = variants_map.get(char, [])
        for idx, start, is_var, sent in data["occurrences"]:
            items.append({
                "char": char,
                "sentence": sent,
                "is_variant": is_var,
                "variants": variants,
                "defs": defs,
                "idx": idx,
                "start": start,
            })
    return items


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split_items_round_robin(items, n_buckets):
    buckets = [[] for _ in range(n_buckets)]
    loads = [0] * n_buckets
    for idx, item in enumerate(items):
        target = idx % n_buckets
        buckets[target].append(item)
        loads[target] += 1

    return buckets, loads


def build_worker_specs(parts, loads, gpu_count, n_models_parallel):
    specs = []
    for gpu_id in range(gpu_count):
        for model_slot in range(n_models_parallel):
            worker_idx = gpu_id * n_models_parallel + model_slot
            part = parts[worker_idx]
            if not part:
                continue
            specs.append({
                "worker_idx": worker_idx,
                "worker_key": f"gpu{gpu_id}_model{model_slot}",
                "gpu_id": gpu_id,
                "model_slot": model_slot,
                "planned_occurrences": len(part),
                "assigned_load": loads[worker_idx],
                "items": part,
            })
    return specs


def snapshot_processes(proc_meta):
    return {
        meta["worker_key"]: {
            "worker_key": meta["worker_key"],
            "gpu_id": meta["gpu_id"],
            "model_slot": meta["model_slot"],
            "pid": meta["process"].pid,
            "exitcode": meta["process"].exitcode,
            "planned_occurrences": meta["planned_occurrences"],
            "output_path": meta["output_path"],
        }
        for meta in proc_meta
    }


def build_summary(total, completed, t0, run_meta, gpu_stats, proc_meta):
    now = time.time()
    worker_entries = {}
    per_gpu = {}
    total_gpu_time = 0.0
    inflight_gpu_time = 0.0

    for worker_key, value in dict(gpu_stats).items():
        entry = dict(value)
        active_started_at = entry.get("active_batch_started_at")
        if active_started_at:
            active_elapsed = max(0.0, now - active_started_at)
        else:
            active_elapsed = 0.0

        entry["active_batch_elapsed_sec"] = active_elapsed
        worker_entries[worker_key] = entry
        total_gpu_time += float(entry.get("total_time_sec", 0.0))
        inflight_gpu_time += active_elapsed

        gpu_key = str(entry["gpu_id"])
        gpu_entry = per_gpu.setdefault(gpu_key, {
            "gpu_id": entry["gpu_id"],
            "planned_occurrences": 0,
            "completed_occurrences": 0,
            "completed_batches": 0,
            "total_batches": 0,
            "total_time_sec": 0.0,
            "active_batch_elapsed_sec": 0.0,
            "worker_keys": [],
            "statuses": {},
        })
        gpu_entry["planned_occurrences"] += int(entry.get("planned_occurrences", 0))
        gpu_entry["completed_occurrences"] += int(entry.get("completed_occurrences", 0))
        gpu_entry["completed_batches"] += int(entry.get("completed_batches", 0))
        gpu_entry["total_batches"] += int(entry.get("total_batches", 0))
        gpu_entry["total_time_sec"] += float(entry.get("total_time_sec", 0.0))
        gpu_entry["active_batch_elapsed_sec"] += active_elapsed
        gpu_entry["worker_keys"].append(worker_key)
        gpu_entry["statuses"][worker_key] = entry.get("status")

    for entry in per_gpu.values():
        completed_occurrences = entry["completed_occurrences"]
        entry["avg_sec_per_occ"] = (
            entry["total_time_sec"] / completed_occurrences
        ) if completed_occurrences else 0.0

    return {
        "run": {
            **run_meta,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "ALL": {
            "total_occurrences": total,
            "completed_occurrences": completed,
            "pending_occurrences": max(total - completed, 0),
            "wall_time_sec": now - t0,
            "gpu_time_sec_completed": total_gpu_time,
            "gpu_time_sec_inflight_estimate": inflight_gpu_time,
            "avg_sec_per_occ_weighted": (total_gpu_time / completed) if completed else 0.0,
        },
        "per_worker": worker_entries,
        "per_gpu": per_gpu,
        "processes": snapshot_processes(proc_meta),
    }


def write_progress_summary(summary_paths, total, counter, t0, run_meta, gpu_stats, proc_meta):
    completed = counter.value
    summary = build_summary(total, completed, t0, run_meta, gpu_stats, proc_meta)
    atomic_write_json(summary_paths["run"], summary)

    alias_path = summary_paths.get("alias")
    if alias_path is not None and alias_path != summary_paths["run"]:
        atomic_write_json(alias_path, summary)

    return summary


def run_worker(worker_key, phys_gpu_id, model_slot, total_models_on_gpu, items, cfg, out_path,
               counter=None, gpu_stats=None, pbar_position=0,
               system_prompt=DEFAULT_SYSTEM_PROMPT):
    batch_size = cfg["batch_size"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_gpu_id)
    os.environ["VLLM_WORKER_USE_RAY"] = "0"
    per_model_gpu_memory_utilization = cfg.get(
        "gpu_memory_utilization",
        min(0.9 / max(total_models_on_gpu, 1), 0.9),
    )

    planned_occurrences = len(items)
    total_time = 0.0
    total_occ = 0
    batches_completed = 0
    active_batch_started_at = None
    pbar = None

    def publish(status, extra=None):
        if gpu_stats is None:
            return

        payload = {
            "worker_key": worker_key,
            "gpu_id": phys_gpu_id,
            "model_slot": model_slot,
            "models_parallel_on_gpu": total_models_on_gpu,
            "planned_occurrences": planned_occurrences,
            "completed_occurrences": total_occ,
            "completed_batches": batches_completed,
            "total_batches": (planned_occurrences + batch_size - 1) // batch_size if planned_occurrences else 0,
            "total_time_sec": total_time,
            "avg_sec_per_occ": (total_time / total_occ) if total_occ else 0.0,
            "status": status,
            "active_batch_started_at": active_batch_started_at,
            "gpu_memory_utilization": per_model_gpu_memory_utilization,
        }
        if extra:
            payload.update(extra)
        gpu_stats[worker_key] = payload

    publish("initializing")

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
        llm = LLM(
            model=cfg["model_path"],
            tensor_parallel_size=1,
            gpu_memory_utilization=per_model_gpu_memory_utilization,
        )

        sp = SamplingParams(
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            repetition_penalty=cfg["repetition_penalty"],
            max_tokens=cfg["max_tokens"],
        )

        with open(out_path, "w", encoding="utf-8") as fout:
            pbar = tqdm(
                total=planned_occurrences,
                desc=f"GPU-{phys_gpu_id}/M-{model_slot}",
                dynamic_ncols=True,
                position=pbar_position,
                leave=True,
            )

            for batch in chunks(items, batch_size):
                active_batch_started_at = time.time()
                publish("running", {"active_batch_size": len(batch)})

                prompts1, metas = [], []
                for it in batch:
                    char = it["char"]
                    variants = it.get("variants", [])
                    char_with_variants = char if not it["is_variant"] else f'{char}，又写作"{ " ".join(variants) }"'
                    defs = [f"{i + 1}. {d}" for i, d in enumerate(it["defs"])]
                    def_list = "\n".join(defs)

                    p1_text = (
                        f"{char_with_variants}。\n该字有{len(defs)}种释义：\n{def_list}"
                        f'\n在以下句中"{char}"接近以上第几个释义，简要说明原因和答案编号（若以上释义均不准确，回答编号0）：\n{it["sentence"]}'
                    )
                    messages1 = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p1_text},
                    ]
                    chat1 = tokenizer.apply_chat_template(
                        messages1, tokenize=False, add_generation_prompt=True
                    )
                    prompts1.append(chat1)
                    metas.append(it)

                resp1 = llm.generate(prompts1, sampling_params=sp)

                prompts2 = []
                for r1 in resp1:
                    r1_text = r1.outputs[0].text.strip().replace("\n", "")
                    p2_text = f"以下句中，最终选择了答案编号几，仅回答那一个编号数字：\n{r1_text}"
                    messages2 = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p2_text},
                    ]
                    chat2 = tokenizer.apply_chat_template(
                        messages2, tokenize=False, add_generation_prompt=True
                    )
                    prompts2.append(chat2)

                resp2 = llm.generate(prompts2, sampling_params=sp)

                elapsed = time.time() - active_batch_started_at
                total_time += elapsed
                total_occ += len(batch)
                batches_completed += 1
                active_batch_started_at = None

                avg_time = total_time / total_occ
                pbar.set_postfix_str(f"avg {avg_time:.3f}s/occ")
                pbar.update(len(batch))

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

                fout.flush()

                if counter is not None:
                    with counter.get_lock():
                        counter.value += len(batch)

                publish("running")

        publish("completed")
    except Exception as exc:
        if active_batch_started_at is not None:
            total_time += max(0.0, time.time() - active_batch_started_at)
            active_batch_started_at = None
        publish("failed", {"error": repr(exc)})
        raise
    finally:
        if pbar is not None:
            pbar.close()


def main(cfg_path="config.yaml"):
    mp.set_start_method("spawn", force=True)

    resolved_cfg_path = resolve_config_path(cfg_path)
    with open(resolved_cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    prompt_cfg = cfg["prompts"]
    cfg_dir = resolved_cfg_path.parent

    model_path = resolve_model_path(model_cfg["model_path"], Path.cwd(), cfg_dir, SCRIPT_DIR)
    prompt_path = resolve_input_path(prompt_cfg["prompt_path"], Path.cwd(), cfg_dir, SCRIPT_DIR)
    pair_path = resolve_input_path(prompt_cfg["pair_path"], Path.cwd(), cfg_dir, SCRIPT_DIR)
    variant_path = resolve_input_path(prompt_cfg["variant_path"], Path.cwd(), cfg_dir, SCRIPT_DIR)
    progress_alias_path = resolve_output_path(prompt_cfg["progress_path"])

    model_cfg = deepcopy(model_cfg)
    model_cfg["model_path"] = model_path

    items = flatten_prompts(prompt_path, pair_path, variant_path)

    gpu_count = torch.cuda.device_count()
    assert gpu_count >= 1, "No GPUs visible"
    n_models_parallel = max(1, int(model_cfg.get("n_models_parallel", 1)))
    total_worker_slots = gpu_count * n_models_parallel

    parts, loads = split_items_round_robin(items, total_worker_slots)
    worker_specs = build_worker_specs(parts, loads, gpu_count, n_models_parallel)
    per_gpu_assignment_occurrences = {
        str(gpu_id): sum(spec["planned_occurrences"] for spec in worker_specs if spec["gpu_id"] == gpu_id)
        for gpu_id in range(gpu_count)
        if any(spec["gpu_id"] == gpu_id for spec in worker_specs)
    }
    per_gpu_worker_keys = {
        str(gpu_id): [spec["worker_key"] for spec in worker_specs if spec["gpu_id"] == gpu_id]
        for gpu_id in range(gpu_count)
        if any(spec["gpu_id"] == gpu_id for spec in worker_specs)
    }
    per_worker_assignment_occurrences = {
        spec["worker_key"]: spec["planned_occurrences"] for spec in worker_specs
    }

    run_dir = make_run_dir(model_cfg, prompt_cfg, progress_alias_path)
    run_summary_path = run_dir / progress_alias_path.name
    config_copy_path = run_dir / resolved_cfg_path.name

    runtime_cfg = deepcopy(cfg)
    runtime_cfg["_runtime"] = {
        "config_path": str(resolved_cfg_path),
        "prompt_path_resolved": str(prompt_path),
        "pair_path_resolved": str(pair_path),
        "variant_path_resolved": str(variant_path),
        "summary_path": str(run_summary_path),
        "summary_alias_path": str(progress_alias_path),
        "output_dir": str(run_dir),
        "n_models_parallel": n_models_parallel,
        "worker_assignment_occurrences": per_worker_assignment_occurrences,
        "gpu_assignment_occurrences": per_gpu_assignment_occurrences,
        "gpu_worker_keys": per_gpu_worker_keys,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_yaml(config_copy_path, runtime_cfg)

    counter = mp.Value("i", 0)
    stop_event = threading.Event()
    pbar_total = tqdm(
        total=len(items),
        desc="ALL",
        dynamic_ncols=True,
        position=len(worker_specs),
        leave=True,
    )

    run_meta = {
        "config_path": str(resolved_cfg_path),
        "config_copy_path": str(config_copy_path),
        "output_dir": str(run_dir),
        "summary_path": str(run_summary_path),
        "summary_alias_path": str(progress_alias_path),
        "model_path": str(model_cfg["model_path"]),
        "prompt_path": str(prompt_path),
        "pair_path": str(pair_path),
        "variant_path": str(variant_path),
        "model_settings": {
            "batch_size": model_cfg["batch_size"],
            "n_models_parallel": n_models_parallel,
            "temperature": model_cfg["temperature"],
            "top_p": model_cfg["top_p"],
            "repetition_penalty": model_cfg["repetition_penalty"],
            "max_tokens": model_cfg["max_tokens"],
            "gpu_memory_utilization_per_model": model_cfg.get(
                "gpu_memory_utilization",
                min(0.9 / max(n_models_parallel, 1), 0.9),
            ),
        },
        "worker_assignment_occurrences": per_worker_assignment_occurrences,
        "gpu_assignment_occurrences": per_gpu_assignment_occurrences,
        "gpu_worker_keys": per_gpu_worker_keys,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_paths = {"run": run_summary_path, "alias": progress_alias_path}

    try:
        with Manager() as mgr:
            gpu_stats = mgr.dict()
            t0 = time.time()
            proc_meta = []

            for bar_position, spec in enumerate(worker_specs):
                out_path = run_dir / f"out_gpu{spec['gpu_id']}_model{spec['model_slot']}.jsonl"
                proc = mp.Process(
                    target=run_worker,
                    args=(
                        spec["worker_key"],
                        spec["gpu_id"],
                        spec["model_slot"],
                        n_models_parallel,
                        spec["items"],
                        model_cfg,
                        str(out_path),
                    ),
                    kwargs={
                        "counter": counter,
                        "gpu_stats": gpu_stats,
                        "pbar_position": bar_position,
                    },
                )
                proc.start()
                proc_meta.append({
                    "worker_key": spec["worker_key"],
                    "gpu_id": spec["gpu_id"],
                    "model_slot": spec["model_slot"],
                    "planned_occurrences": spec["planned_occurrences"],
                    "output_path": str(out_path),
                    "process": proc,
                })

            write_progress_summary(summary_paths, len(items), counter, t0, run_meta, gpu_stats, proc_meta)

            def _monitor():
                last_counter = -1
                last_summary_at = 0.0
                while not stop_event.is_set():
                    current = counter.value
                    if current != last_counter:
                        pbar_total.n = current
                        pbar_total.refresh()
                        last_counter = current

                    now = time.time()
                    if now - last_summary_at >= SUMMARY_WRITE_INTERVAL_SEC:
                        write_progress_summary(summary_paths, len(items), counter, t0, run_meta, gpu_stats, proc_meta)
                        last_summary_at = now

                    time.sleep(0.1)

            monitor = threading.Thread(target=_monitor, daemon=True)
            monitor.start()

            for meta in proc_meta:
                meta["process"].join()

            stop_event.set()
            monitor.join(timeout=1.0)

            summary = write_progress_summary(summary_paths, len(items), counter, t0, run_meta, gpu_stats, proc_meta)

            failed = [meta for meta in proc_meta if meta["process"].exitcode not in (0, None)]
            if failed:
                failed_workers = ", ".join(
                    f"{meta['worker_key']} (exitcode={meta['process'].exitcode})"
                    for meta in failed
                )
                raise RuntimeError(f"One or more worker processes failed: {failed_workers}")

            return summary
    finally:
        pbar_total.n = counter.value
        pbar_total.refresh()
        pbar_total.close()


if __name__ == "__main__":
    main()
