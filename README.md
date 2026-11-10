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
gpu/combos_gpu.py           inferência na GPU com frames, tokens e modo configuráveis (inclui o modo direto R5)
benchmark/                  tempo, memória e comparação das configurações R1–R7 (ver benchmark/README.md)
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

### Modo direto (R5): mais rápido e com AUC maior

O prompt já termina em `<answer>`, então o modelo gera **1 token só**, sem escrever o raciocínio, e o
P(fake) sai direto da probabilidade desse token. O vídeo é fake quando a média de P(fake) dos 3 clipes
passa de 0,5.

```bash
VV_MODEL=checkpoints/VideoVeritas python gpu/combos_gpu.py --nome R5_14f_768t_direto \
       --frames 14 --tokens 768 --modo direto          # → benchmark/combos/R5_14f_768t_direto.jsonl
```

| | AUC [IC95] | Reais certos | Fakes certos | GPU s/vídeo | CPU min/vídeo |
|---|---|---|---|---|---|
| protocolo original (R1, com raciocínio) | 0,898 [0,839–0,951] | 31/35 | 81/95 | 7,9 | 15,7 |
| **modo direto (R5)** | **0,945 [0,904–0,976]** | 33/35 | 75/95 | 2,6 | ~4,6 (estimado) |

O ganho de AUC foi significativo: ΔAUC +0,047 [+0,009; +0,091], por bootstrap pareado em 130 vídeos.
O modo direto não produz justificativa em texto, e o limiar 0,5 acerta menos fakes, então convém
recalibrá-lo. O mesmo script roda as outras configurações (`--frames`, `--tokens`,
`--modo raciocinio|direto`). Detalhes em [benchmark/README.md](benchmark/README.md).

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

Custos no i9-14900K e na RTX 4090, por vídeo (3 clipes), com raciocínio:

| | GPU (vLLM, bf16) | CPU (transformers, fp32, 24 threads) |
|---|---|---|
| 14 frames por clipe (protocolo) | 9,8 s em lote; 6,9–16,7 s um vídeo por vez | **14,6 min** (199–455 s por clipe: ~110 s de leitura da entrada + ~0,7 s por token gerado) |
| 26 frames por clipe | 12,3 s em lote; 7,6–22,0 s um vídeo por vez | ~16 min (286–335 s por clipe: ~190 s de leitura da entrada + ~1 s por token gerado) |
| memória | 16,6 GB de pesos + cache (o vLLM reserva ~21 GB) | 35–40 GB de RAM |

Todos os valores foram medidos com `benchmark/bench.py`: 6 vídeos na GPU e 1 vídeo (3 clipes) na CPU. O
modelo é carregado uma vez (~44 s na GPU), e esse tempo não entra na tabela. Os vereditos na CPU foram iguais
aos da GPU. Na CPU o tempo depende muito do tamanho do raciocínio gerado, que vai de ~120 a ~480 tokens por clipe.

A CPU roda em fp32 porque processadores sem AVX-512 BF16/AMX não fazem bf16 nativo. Os vereditos na CPU
foram iguais aos da GPU.

## Créditos

VideoVeritas: Tan et al., [arXiv:2602.08828](https://arxiv.org/abs/2602.08828), código em
[EricTan7/VideoVeritas](https://github.com/EricTan7/VideoVeritas) (Apache-2.0).
