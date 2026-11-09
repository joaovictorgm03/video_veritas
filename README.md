# video_veritas

Scripts para rodar o [VideoVeritas](https://github.com/EricTan7/VideoVeritas) (Qwen3-VL-8B ajustado para
detectar vídeo real × gerado por IA) em **GPU** (vLLM) e em **CPU** (transformers), usados na avaliação
sobre um conjunto de vídeos do TikTok.

Cada vídeo é cortado em **3 clipes de 5 s** (centrados em 20 %, 50 % e 80 % da duração) e cada clipe é
amostrado a 3 fps, o que dá 14 frames. O modelo responde `real` ou `fake` dentro de `<answer>…</answer>`.
O vídeo é `fake` quando ≥ 2 dos 3 clipes dizem `fake`, e o score do vídeo é a média de P(fake) dos clipes.

```
comum/04_clipes.py          corta os 3 clipes de 5 s de cada vídeo (ffmpeg) → data/clips.csv
gpu/05_inferir_video.py     inferência em lote na GPU (vLLM), com P(fake) pelos logprobs → .jsonl
benchmark/bench.py          tempo por vídeo na GPU (vLLM) ou na CPU (transformers, fp32)
third_party/VideoVeritas/   qwen_vl_utils do repositório oficial (Apache-2.0, sem alterações)
```

## Instalação

```bash
conda create -n veritas python=3.11 -y && conda activate veritas
pip install -r requirements-gpu.txt      # GPU
# pip install -r requirements-cpu.txt    # só CPU (sem vLLM)

# pesos (~17 GB, bf16), em checkpoints/VideoVeritas
pip install modelscope
modelscope download --model EricTanh/VideoVeritas --local_dir checkpoints/VideoVeritas
```

O `ffmpeg` também precisa estar instalado no sistema.

## 1. Cortar os clipes (comum)

São necessários dois arquivos de entrada:
- `data/videos.csv`, com as colunas `video_id,label,file,duration`, onde `label` vale `real` ou `fake` e `duration` está em segundos;
- os vídeos, em `<dataset>/<label>/<file>`.

```bash
python comum/04_clipes.py --dataset /caminho/do/dataset --data data
```

## 2. GPU

```bash
python gpu/05_inferir_video.py --clips data/clips.csv --out results/predicoes_video.jsonl \
       --model checkpoints/VideoVeritas
# GPU com pouca memória livre (ex.: 24 GB com outros processos):
GPU_UTIL=0.88 MAX_LEN=12288 python gpu/05_inferir_video.py ...
```

Rode a partir da raiz do repositório, porque os caminhos de `clips.csv` são relativos a ela. O script retoma
de onde parou se o `.jsonl` já existir. `--limit N` processa só N clipes, para teste.

## 3. CPU

```bash
python benchmark/bench.py cpu --frames 14 --videos 1 --clips 3 --threads 24   # 1 vídeo completo = 3 clipes
python benchmark/bench.py gpu --frames 14 26 --videos 5             # mesma medição na GPU
```

Variáveis de ambiente:
- `VV_MODEL`: pesos; o padrão é `checkpoints/VideoVeritas`;
- `VV_DATA`: pasta que contém `data/clips.csv`; o padrão é a pasta atual;
- `VV_REPO`: pasta com `swift/qwen_vl_utils`; o padrão é `third_party/VideoVeritas`.

O resultado vai para `benchmark/resultados_<cpu|gpu>.json`.

Custos medidos (i9-14900K e RTX 4090, protocolo de 14 frames com raciocínio):

| | GPU | CPU (fp32, 24 threads) |
|---|---|---|
| tempo por vídeo (3 clipes) | ~10–14 s | ~16 min |
| memória | 16,6 GB de pesos + cache (o vLLM reserva ~21 GB) | ~35–40 GB de RAM |

A CPU roda em fp32 porque processadores sem AVX-512 BF16/AMX não fazem bf16 nativo. Os vereditos na CPU
foram iguais aos da GPU.

## Créditos

VideoVeritas: Tan et al., [arXiv:2602.08828](https://arxiv.org/abs/2602.08828), código em
[EricTan7/VideoVeritas](https://github.com/EricTan7/VideoVeritas) (Apache-2.0).
