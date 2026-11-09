"""Corta 3 clipes de 5 s (20%, 50%, 80% da duração) de cada vídeo do TikTok, para o VideoVeritas.

Os clipes são re-encodados em h264 CRF 10 (praticamente sem perda): corte preciso fora de keyframe
e, de quebra, todos os clipes ficam no mesmo codec, o que tira o atalho h264/hevc da entrada.
Saída: data/clips/<video_id>_c<k>.mp4 e data/clips.csv
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd

CLIP_SECONDS = 5.0
POSITIONS = [ 0.2, 0.5, 0.8 ]


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required = True)
	parser.add_argument('--data', required = True)
	args = parser.parse_args()

	dataset, data = Path(args.dataset), Path(args.data)
	videos = pd.read_csv(data / 'videos.csv')
	(data / 'clips').mkdir(exist_ok = True)
	rows = []

	for v in videos.itertuples():
		src = dataset / v.label / v.file
		for k, pos in enumerate(POSITIONS):
			start = max(0.0, min(v.duration * pos - CLIP_SECONDS / 2, v.duration - CLIP_SECONDS))
			dst = data / 'clips' / f'{v.video_id}_c{k}.mp4'
			if not dst.exists():
				subprocess.run([ 'ffmpeg', '-v', 'error', '-y', '-ss', f'{start:.3f}', '-i', str(src), '-t', str(CLIP_SECONDS), '-an', '-c:v', 'libx264', '-crf', '10', '-preset', 'fast', '-pix_fmt', 'yuv420p', str(dst) ], check = True)
			rows.append({ 'video_id': v.video_id, 'label': v.label, 'clip': k, 'start': round(start, 3), 'path': str(dst) })
		print(v.video_id, flush = True)

	pd.DataFrame(rows).to_csv(data / 'clips.csv', index = False)


if __name__ == '__main__':
	main()
