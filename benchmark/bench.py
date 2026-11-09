"""Benchmark de tempo do VideoVeritas (Qwen3-VL-8B) por vídeo = 3 clipes de 5 s.

Mesmo prompt, mesmo pré-processamento (qwen_vl_utils do repositório, <= 768 tokens por par de frames,
do_resize=False) e mesmos clipes da avaliação (data/clips.csv, gerado por comum/04_clipes.py).
Frames por clipe: o Qwen3-VL agrupa frames de 2 em 2 (FRAME_FACTOR=2), então 25 não é possível:
usamos 26 (pedido: 25) e 14 (o que o protocolo atual 3 fps × 5 s realmente usa).

  GPU: vLLM offline (como 05_inferir_video.py), temperatura 0.
  CPU: transformers, fp32 (o i9-14900K não tem AVX-512/AMX para bf16), temperatura 0.

Mede separadamente: carga do modelo, pré-processamento (decodificar + redimensionar frames),
inferência (prefill + geração). Saída: resultados_<modo>.json
Uso: python benchmark/bench.py gpu --frames 14 26 --videos 5
     python benchmark/bench.py cpu --frames 14 --videos 1 --clips 3   # 1 vídeo completo = 3 clipes de 5 s
"""
import argparse
import importlib.util
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("DURATION", "5.0")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import pandas as pd  # noqa: E402

# caminhos configuráveis por variável de ambiente (padrões relativos à raiz deste repositório)
ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("VV_REPO", ROOT / "third_party" / "VideoVeritas"))  # contém swift/qwen_vl_utils/
MODEL = os.environ.get("VV_MODEL", str(ROOT / "checkpoints" / "VideoVeritas"))   # pesos baixados do ModelScope
DATA = Path(os.environ.get("VV_DATA", Path.cwd()))  # pasta onde está data/clips.csv (caminhos dos clipes relativos a ela)
SYSTEM_PROMPT = 'You are an expert video analyst.\nPlease think about the question as if you were a human pondering deeply. It’s encouraged to include self-reflection or verification in the reasoning process. Then, give a final verdict within <answer> </answer> tags.'
USER_PROMPT = 'Is this video real or fake?'
HERE = Path(__file__).parent


def load_vp():
    spec = importlib.util.spec_from_file_location("vision_process", REPO / "swift" / "qwen_vl_utils" / "vision_process.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def message(path, nframes):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "video", "video": str(DATA / path), "nframes": nframes},
                                         {"type": "text", "text": USER_PROMPT}]}]


def answer(text):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
    return m.group(1).strip().lower() if m else None


def pick_videos(n):
    c = pd.read_csv(DATA / "data" / "clips.csv")
    vids = c.drop_duplicates("video_id")
    # metade real, metade fake, ordem fixa
    sel = pd.concat([vids[vids.label == "real"].head((n + 1) // 2), vids[vids.label == "fake"].head(n // 2)]).video_id
    return [c[c.video_id == v].sort_values("clip") for v in sel]


def bench_gpu(args, vp, processor):
    from vllm import LLM, SamplingParams
    t = time.perf_counter()
    llm = LLM(model=MODEL, max_model_len=15000, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.88")),
              limit_mm_per_prompt={"image": 0, "video": 1}, max_num_seqs=8)
    load_s = time.perf_counter() - t
    sp = SamplingParams(temperature=0, max_tokens=2048)
    out = {"modo": "gpu", "carga_modelo_s": load_s, "por_frames": {}}
    vids = pick_videos(args.videos)
    for nf in args.frames:
        def prep(clips):
            t0 = time.perf_counter(); inputs = []
            for p in clips.path:
                msg = message(p, nf)
                text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                _, videos, kw = vp.process_vision_info(msg, image_patch_size=processor.image_processor.patch_size,
                                                       return_video_kwargs=True, return_video_metadata=True)
                kw["do_resize"] = False
                inputs.append({"prompt": text, "multi_modal_data": {"video": videos}, "mm_processor_kwargs": kw})
            return inputs, time.perf_counter() - t0
        warm, _ = prep(vids[0].head(1)); llm.generate(warm, sp, use_tqdm=False)  # aquecimento
        per_video = []
        for clips in vids:  # latência: 1 vídeo (3 clipes) por vez
            inputs, tp = prep(clips)
            t0 = time.perf_counter(); res = llm.generate(inputs, sp, use_tqdm=False); ti = time.perf_counter() - t0
            per_video.append({"video_id": clips.video_id.iloc[0], "label": clips.label.iloc[0], "preproc_s": tp, "inferencia_s": ti,
                              "tokens_prompt": [len(r.prompt_token_ids) for r in res], "tokens_saida": [len(r.outputs[0].token_ids) for r in res],
                              "respostas": [answer(r.outputs[0].text) for r in res]})
            print(f"[gpu {nf} frames] {per_video[-1]['video_id']}: preproc {tp:.1f}s  inferência {ti:.1f}s  "
                  f"prompt {per_video[-1]['tokens_prompt']}  saída {per_video[-1]['tokens_saida']}  {per_video[-1]['respostas']}", flush=True)
        allin, tp_all = [], 0  # vazão: todos os vídeos num lote só
        for clips in vids:
            i, tp = prep(clips); allin += i; tp_all += tp
        t0 = time.perf_counter(); llm.generate(allin, sp, use_tqdm=False); ti_all = time.perf_counter() - t0
        out["por_frames"][nf] = {"latencia_por_video": per_video, "lote": {"videos": len(vids), "preproc_s": tp_all, "inferencia_s": ti_all}}
        print(f"[gpu {nf} frames] lote de {len(vids)} vídeos: preproc {tp_all:.1f}s  inferência {ti_all:.1f}s "
              f"→ {(tp_all + ti_all) / len(vids):.1f} s/vídeo", flush=True)
    return out


def bench_cpu(args, vp, processor):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    torch.set_num_threads(args.threads)
    t = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    load_s = time.perf_counter() - t
    out = {"modo": "cpu", "threads": args.threads, "carga_modelo_s": load_s, "por_frames": {}}
    print(f"[cpu] modelo carregado em {load_s:.0f}s", flush=True)
    for nf in args.frames:
        res = []
        for clips in pick_videos(args.videos):
            for p in clips.path.head(args.clips):
                t0 = time.perf_counter()
                msg = message(p, nf)
                text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                _, videos, kw = vp.process_vision_info(msg, image_patch_size=processor.image_processor.patch_size,
                                                       return_video_kwargs=True, return_video_metadata=True)
                videos, metas = zip(*videos) if isinstance(videos[0], tuple) else (videos, None)
                inp = processor(text=[text], videos=list(videos), video_metadata=list(metas) if metas else None,
                                do_resize=False, return_tensors="pt", **{k: v for k, v in kw.items() if k != "do_resize"})
                tp = time.perf_counter() - t0
                with torch.inference_mode():
                    t1 = time.perf_counter(); model.generate(**inp, max_new_tokens=1, do_sample=False); tf = time.perf_counter() - t1
                    t2 = time.perf_counter(); g = model.generate(**inp, max_new_tokens=2048, do_sample=False); tg = time.perf_counter() - t2
                n_out = g.shape[1] - inp["input_ids"].shape[1]
                txt = processor.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                r = {"video_id": clips.video_id.iloc[0], "clip": p, "preproc_s": tp, "primeiro_token_s": tf, "geracao_total_s": tg,
                     "tokens_prompt": int(inp["input_ids"].shape[1]), "tokens_saida": int(n_out), "resposta": answer(txt)}
                res.append(r)
                print(f"[cpu {nf} frames] {p}: preproc {tp:.1f}s  1º token {tf:.0f}s  total {tg:.0f}s  "
                      f"prompt {r['tokens_prompt']}  saída {n_out}  → {r['resposta']}", flush=True)
                json.dump(out | {"por_frames": out["por_frames"] | {nf: res}}, open(HERE / "resultados_cpu.json", "w"), indent=1)
        out["por_frames"][nf] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["gpu", "cpu"])
    ap.add_argument("--frames", type=int, nargs="+", default=[14, 26])
    ap.add_argument("--videos", type=int, default=5)
    ap.add_argument("--clips", type=int, default=3, help="clipes por vídeo na CPU (o protocolo usa os 3: 20%%, 50%% e 80%% da duração)")
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL)
    vp = load_vp()
    out = bench_gpu(a, vp, processor) if a.modo == "gpu" else bench_cpu(a, vp, processor)
    json.dump(out, open(HERE / f"resultados_{a.modo}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
