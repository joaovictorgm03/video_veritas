#!/bin/bash
# Roda as 7 configurações em sequência (cada uma num processo próprio). Retoma de onde parou.
# Ative antes o ambiente (requirements-gpu.txt) e rode da pasta que contém data/clips.csv:
#   bash benchmark/rodar_combos.sh
# Se faltar memória na GPU, reduza a concorrência: MAX_SEQS=8 (ou 2 para 26 frames) e --batch 16 (ou 4).
export VV_DATA="${VV_DATA:-$PWD}"
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" GPU_UTIL="${GPU_UTIL:-0.88}"
mkdir -p combos
# diretas primeiro (rápidas), depois as com raciocínio
python ../gpu/combos_gpu.py --nome R5_14f_768t_direto     --frames 14 --tokens 768 --modo direto     >> combos/log.txt 2>&1
python ../gpu/combos_gpu.py --nome R6_14f_384t_direto     --frames 14 --tokens 384 --modo direto     >> combos/log.txt 2>&1
python ../gpu/combos_gpu.py --nome R7_14f_256t_direto     --frames 14 --tokens 256 --modo direto     >> combos/log.txt 2>&1
python ../gpu/combos_gpu.py --nome R1_14f_768t_raciocinio --frames 14 --tokens 768 --modo raciocinio >> combos/log.txt 2>&1
python ../gpu/combos_gpu.py --nome R3_14f_384t_raciocinio --frames 14 --tokens 384 --modo raciocinio >> combos/log.txt 2>&1
python ../gpu/combos_gpu.py --nome R4_14f_256t_raciocinio --frames 14 --tokens 256 --modo raciocinio >> combos/log.txt 2>&1
python ../gpu/combos_gpu.py --nome R2_26f_768t_raciocinio --frames 26 --tokens 768 --modo raciocinio >> combos/log.txt 2>&1
echo FIM >> combos/log.txt
