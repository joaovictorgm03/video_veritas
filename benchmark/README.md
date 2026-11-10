# Benchmarks do VideoVeritas

Avaliação sobre **130 vídeos do TikTok** (35 reais e 95 gerados por IA), cada um cortado em **3 clipes de 5 s**
(390 clipes). Hardware: RTX 4090 (24 GB) e i9-14900K (24 threads, 125 GB de RAM).

| Script | O que faz | Saída |
|---|---|---|
| `bench.py` | tempo por vídeo na GPU (vLLM) ou na CPU (transformers, fp32) | `resultados_<gpu\|cpu>.json` |
| `mem_cpu.py` | pico de RAM na CPU em 1 clipe, para 3 configurações | `mem_cpu.json` |
| `../gpu/combos_gpu.py` | roda uma configuração (frames, tokens, modo) nos 390 clipes | `combos/<nome>.jsonl` |
| `rodar_combos.sh` | roda as 7 configurações R1–R7 em sequência | `combos/*.jsonl` |
| `analisar_combos.py` | compara as configurações com a base R1 (AUC, acurácia, tempo) | `combos_resultado.csv` |

`resultados/` guarda os resultados das rodadas publicadas. As saídas novas dos scripts ficam fora do git.

## 1. Tempo por vídeo (3 clipes, com raciocínio)

Fonte: `resultados/resultados_gpu.json`, `resultados_cpu_14f.json` e `resultados_cpu_26f.json`.

| Frames por clipe | GPU, em lote | GPU, um vídeo por vez | CPU (fp32, 24 threads) |
|---|---|---|---|
| 14 (protocolo) | 9,8 s | 6,9–16,7 s | **14,6 min** (199–455 s por clipe) |
| 26 | 12,3 s | 7,6–22,0 s | ~16 min (286–335 s por clipe) |

Na CPU, a leitura da entrada leva ~110 s com 14 frames e ~190 s com 26. Cada token do raciocínio custa mais
~0,7–1 s, e o raciocínio vai de ~120 a ~480 tokens por clipe. A carga do modelo (~44 s na GPU) não entra na
conta. Os vereditos na CPU foram iguais aos da GPU.

## 2. Memória na CPU (fp32)

Fonte: `resultados/mem_cpu.json`.

| Configuração | Tokens de entrada | Pico de RAM |
|---|---|---|
| modelo carregado | — | 33,6 GiB |
| 14 frames, 256 tokens por par de frames | 1.894 | 35,3 GiB |
| 14 frames, 768 tokens por par de frames | 5.170 | 37,4 GiB |
| 26 frames, 768 tokens por par de frames | 9.538 | 40,0 GiB |

## 3. Configurações mais baratas (R1–R7)

Fonte: `resultados/combos_resultado.csv`. Base: R1, o protocolo original, que reproduz a rodada original
com 100 % dos vereditos iguais.

- **Modo raciocínio:** o modelo escreve a análise e o P(fake) é lido no token depois de `<answer>`.
  O vídeo é fake quando ≥ 2 de 3 clipes dizem fake.
- **Modo direto:** o prompt já termina em `<answer>`, então o modelo gera só 1 token. O vídeo é fake quando a
  média de P(fake) passa de 0,5.

| Config. | Frames | Tokens por par | Modo | AUC [IC95] | ΔAUC vs R1 [IC95] | Reais certos | Fakes certos | GPU s/vídeo | CPU min/vídeo* |
|---|---|---|---|---|---|---|---|---|---|
| R1 (base) | 14 | 768 | raciocínio | 0,898 [0,839–0,951] | — | 31/35 | 81/95 | 7,9 | 15,7 |
| R2 | 26 | 768 | raciocínio | 0,924 [0,872–0,965] | +0,026 [−0,020; +0,074] | 31/35 | 82/95 | ~15** | 19,0 |
| R3 | 14 | 384 | raciocínio | 0,888 [0,818–0,948] | −0,011 [−0,057; +0,029] | 27/35 | 88/95 | 4,3 | 12,6 |
| R4 | 14 | 256 | raciocínio | 0,893 [0,824–0,949] | −0,006 [−0,054; +0,033] | 28/35 | 88/95 | 3,5 | 13,1 |
| **R5** | 14 | 768 | **direto** | **0,945 [0,904–0,976]** | **+0,047 [+0,009; +0,091]** | 33/35 | 75/95 | 2,6 | 4,6 |
| R6 | 14 | 384 | direto | 0,937 [0,892–0,971] | +0,038 [−0,007; +0,089] | 31/35 | 78/95 | 1,7 | 2,7 |
| R7 | 14 | 256 | direto | 0,942 [0,897–0,977] | +0,044 [−0,006; +0,100] | 31/35 | 82/95 | 1,3 | 1,9 |

\* Estimado a partir das medições de CPU: ~0,02 s por token de entrada mais ~1 s por token gerado.
\** Inflado: parte da rodada R2 precisou de concorrência 2 por falta de memória na GPU.

A tabela completa também traz as variantes com parada antecipada, só o clipe do meio e só 2 clipes.

**Leitura dos resultados:**
- **R5 foi a única configuração com ganho significativo de AUC**, porque o intervalo do ΔAUC não inclui zero.
  - Após o raciocínio, o P(fake) fica saturado: 100 % dos clipes têm P < 0,01 ou > 0,99, e só aparecem 2 valores distintos. Na prática, ele vira um voto.
  - No modo direto, o P(fake) é graduado (154 valores distintos) e ordena melhor os vídeos.
- **A acurácia com limiar 0,5 fica parecida em todas** (~0,85–0,87 balanceada). O modo direto acerta mais reais e menos fakes, então o ganho está no ranking, e o limiar precisaria ser recalibrado por validação cruzada.
- **No modo direto não há texto de justificativa.**

## Como reproduzir

Rode da pasta que contém `data/clips.csv` (gerado por `comum/04_clipes.py`):

```bash
python benchmark/bench.py gpu --frames 14 26 --videos 6
python benchmark/bench.py cpu --frames 14 --videos 1 --clips 3 --threads 24
python benchmark/mem_cpu.py
bash benchmark/rodar_combos.sh                       # ~1 h 10 na RTX 4090
VV_ORIG=results/predicoes_video.jsonl python benchmark/analisar_combos.py
```

`VV_ORIG` é opcional. Ele aponta para as predições de `gpu/05_inferir_video.py` e serve para conferir se a
base R1 reproduz a rodada original.
