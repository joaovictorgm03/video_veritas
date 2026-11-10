"""Testa na GPU configurações mais baratas do VideoVeritas sobre os 390 clipes (130 vídeos × 3) da avaliação.

Uma configuração = (frames por clipe, tokens máx. por par de frames, modo):
  modo "raciocinio": protocolo original (gera o raciocínio e lê P(fake) após <answer>);
  modo "direto":     o prompt termina com "<answer>" e lê-se P(fake) do próximo token (1 passada, sem gerar texto).
Mesmo prompt de sistema/usuário, mesmo qwen_vl_utils do repositório (do_resize=False), temperatura 0.
Cada configuração roda num processo próprio (VIDEO_MAX_TOKEN_NUM é lido na importação do módulo).
Saída: combos/<nome>.jsonl (uma linha por clipe, com tempos e tokens).
Uso: python gpu/combos_gpu.py --nome R5_14f_768t_direto --frames 14 --tokens 768 --modo direto
     python gpu/combos_gpu.py --nome R1_14f_768t_raciocinio --frames 14 --tokens 768 --modo raciocinio
"""
import argparse
import importlib.util
import json
import math
import os
import re
import time
from pathlib import Path

a0 = argparse.ArgumentParser()
a0.add_argument("--nome", required=True)
a0.add_argument("--frames", type=int, required=True)
a0.add_argument("--tokens", type=int, required=True)
a0.add_argument("--modo", choices=["raciocinio", "direto"], required=True)
a0.add_argument("--batch", type=int, default=30)
ARGS = a0.parse_args()
os.environ["VIDEO_MAX_TOKEN_NUM"] = str(ARGS.tokens)
os.environ.setdefault("DURATION", "5.0")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import pandas as pd  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

# caminhos configuráveis por variável de ambiente (padrões relativos à raiz deste repositório)
ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("VV_REPO", ROOT / "third_party" / "VideoVeritas"))  # contém swift/qwen_vl_utils/
MODEL = os.environ.get("VV_MODEL", str(ROOT / "checkpoints" / "VideoVeritas"))   # pesos baixados do ModelScope
DATA = Path(os.environ.get("VV_DATA", Path.cwd()))  # pasta onde está data/clips.csv (caminhos dos clipes relativos a ela)
SYSTEM_PROMPT = 'You are an expert video analyst.\nPlease think about the question as if you were a human pondering deeply. It’s encouraged to include self-reflection or verification in the reasoning process. Then, give a final verdict within <answer> </answer> tags.'
USER_PROMPT = 'Is this video real or fake?'
OUT = Path(os.environ.get("VV_OUT", ROOT / "benchmark" / "combos"))  # onde gravar <nome>.jsonl


def load_vp():
    spec = importlib.util.spec_from_file_location("vision_process", REPO / "swift" / "qwen_vl_utils" / "vision_process.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.VIDEO_MAX_TOKEN_NUM == ARGS.tokens, (m.VIDEO_MAX_TOKEN_NUM, ARGS.tokens)
    return m


def p_from_step(step):
    p = {"real": 0.0, "fake": 0.0}
    for cand in step.values():
        w = (cand.decoded_token or "").strip().lower()
        if w in p:
            p[w] += math.exp(cand.logprob)
    return p["fake"] / (p["real"] + p["fake"]) if p["real"] + p["fake"] > 0 else None


def prob_after_answer(c):
    """igual ao 05_inferir_video.py: P(fake) no token logo após '<answer>' no texto gerado."""
    if not c.logprobs:
        return None
    text = ""
    for tid, step in zip(c.token_ids, c.logprobs):
        if text.rstrip().endswith("<answer>"):
            return p_from_step(step)
        text += step[tid].decoded_token or ""
    return None


def parse(text):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
    if not m:
        return None
    s = m.group(1).lower()
    return ("real" if "real" in s else "fake") if ("real" in s) != ("fake" in s) else None


def main():
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{ARGS.nome}.jsonl"
    done = {json.loads(line)["path"] for line in out.read_text().splitlines()} if out.exists() else set()
    clips = pd.read_csv(DATA / "data" / "clips.csv")
    items = [r._asdict() for r in clips.itertuples(index=False) if r.path not in done]
    print(f"{ARGS.nome}: frames={ARGS.frames} tokens={ARGS.tokens} modo={ARGS.modo} | {len(items)} clipes a processar", flush=True)
    vp = load_vp()
    proc = AutoProcessor.from_pretrained(MODEL)
    llm = LLM(model=MODEL, max_model_len=15000, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.88")),
              limit_mm_per_prompt={"image": 0, "video": 1}, max_num_seqs=int(os.environ.get("MAX_SEQS", "16")))
    sp = SamplingParams(temperature=0, max_tokens=1 if ARGS.modo == "direto" else 2048, logprobs=20 if ARGS.modo == "direto" else 5)
    with out.open("a") as f:
        for s in range(0, len(items), ARGS.batch):
            batch = items[s:s + ARGS.batch]
            t0 = time.perf_counter(); inputs = []
            for it in batch:
                msg = [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": [{"type": "video", "video": str(DATA / it["path"]), "nframes": ARGS.frames},
                                                    {"type": "text", "text": USER_PROMPT}]}]
                text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                if ARGS.modo == "direto":
                    text += "<answer>"
                _, videos, kw = vp.process_vision_info(msg, image_patch_size=proc.image_processor.patch_size,
                                                       return_video_kwargs=True, return_video_metadata=True)
                kw["do_resize"] = False
                inputs.append({"prompt": text, "multi_modal_data": {"video": videos}, "mm_processor_kwargs": kw})
            tp = time.perf_counter() - t0
            t1 = time.perf_counter(); res = llm.generate(inputs, sp, use_tqdm=False); ti = time.perf_counter() - t1
            for it, r in zip(batch, res):
                c = r.outputs[0]
                if ARGS.modo == "direto":
                    pf = p_from_step(c.logprobs[0]) if c.logprobs else None
                    pred = None if pf is None else ("fake" if pf > 0.5 else "real")
                else:
                    pf, pred = prob_after_answer(c), parse(c.text)
                f.write(json.dumps({**it, "config": ARGS.nome, "frames": ARGS.frames, "tokens_frame": ARGS.tokens, "modo": ARGS.modo,
                                    "pred": pred, "p_fake": pf, "finish": c.finish_reason, "n_prompt": len(r.prompt_token_ids),
                                    "n_out": len(c.token_ids), "t_preproc_lote": tp / len(batch), "t_infer_lote": ti / len(batch),
                                    "text": c.text if ARGS.modo == "raciocinio" else ""}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(s + ARGS.batch, len(items))}/{len(items)}  ({ti / len(batch):.2f} s/clipe)", flush=True)


if __name__ == "__main__":
    main()
