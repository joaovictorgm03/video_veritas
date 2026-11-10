"""Mede o pico de RAM (RSS) do VideoVeritas na CPU (fp32), por configuração, em 1 clipe.
Amostra /proc/self/status a cada 0,2 s numa thread. Só a passada de leitura da entrada + 1 token
(é onde está o pico; gerar mais tokens só acrescenta o KV cache, pequeno)."""
import importlib.util, os, threading, time, json
os.environ.setdefault("DURATION", "5.0")
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("VV_REPO", ROOT / "third_party" / "VideoVeritas"))  # contém swift/qwen_vl_utils/
MODEL = os.environ.get("VV_MODEL", str(ROOT / "checkpoints" / "VideoVeritas"))   # pesos baixados do ModelScope
DATA = Path(os.environ.get("VV_DATA", Path.cwd()))  # pasta onde está data/clips.csv (caminhos dos clipes relativos a ela)
CLIP = os.environ.get("VV_CLIP", "data/clips/real_ad866bbc23_c0.mp4")  # clipe medido (relativo a VV_DATA)
SYS = 'You are an expert video analyst.\nPlease think about the question as if you were a human pondering deeply. It’s encouraged to include self-reflection or verification in the reasoning process. Then, give a final verdict within <answer> </answer> tags.'
def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"): return int(l.split()[1]) / 2**20  # GiB
peak = [0.0]
def mon():
    while True:
        peak[0] = max(peak[0], rss()); time.sleep(0.2)
threading.Thread(target=mon, daemon=True).start()
torch.set_num_threads(24)
base = rss()
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
proc = AutoProcessor.from_pretrained(MODEL)
time.sleep(1); after_load = rss()
out = {"rss_antes_GiB": base, "rss_modelo_carregado_GiB": after_load}
print(out, flush=True)
for nf, tok in [(14, 256), (14, 768), (26, 768)]:
    os.environ["VIDEO_MAX_TOKEN_NUM"] = str(tok)
    spec = importlib.util.spec_from_file_location("vp", REPO / "swift" / "qwen_vl_utils" / "vision_process.py")
    vp = importlib.util.module_from_spec(spec); spec.loader.exec_module(vp)
    msg = [{"role": "system", "content": SYS}, {"role": "user", "content": [{"type": "video", "video": str(DATA / CLIP), "nframes": nf}, {"type": "text", "text": "Is this video real or fake?"}]}]
    text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    _, videos, kw = vp.process_vision_info(msg, image_patch_size=proc.image_processor.patch_size, return_video_kwargs=True, return_video_metadata=True)
    videos, metas = zip(*videos) if isinstance(videos[0], tuple) else (videos, None)
    inp = proc(text=[text], videos=list(videos), video_metadata=list(metas) if metas else None, do_resize=False, return_tensors="pt", **{k: v for k, v in kw.items() if k != "do_resize"})
    peak[0] = rss(); t = time.perf_counter()
    with torch.inference_mode():
        model.generate(**inp, max_new_tokens=1, do_sample=False)
    r = {"frames": nf, "tokens_par": tok, "tokens_entrada": int(inp["input_ids"].shape[1]), "pico_GiB": round(peak[0], 1), "s": round(time.perf_counter() - t)}
    out[f"{nf}f_{tok}t"] = r; print(r, flush=True)
json.dump(out, open(Path(__file__).parent / "mem_cpu.json", "w"), indent=1)
