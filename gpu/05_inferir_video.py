"""Roda o VideoVeritas (Qwen3-VL-8B) offline no vLLM sobre os clipes de 5 s.

Espelha VideoVeritas/self_scripts/infer/infer_vllm.py: mesmo prompt, FPS=3, DURATION=5, leitura e resize dos
frames pelo qwen_vl_utils do próprio repositório (<= 768 tokens por frame) e do_resize=False no processador.
Diferença: temperatura 0 (o oficial usa 0,7) e logprobs para tirar P(fake) do token depois de <answer>.
Saída: results/predicoes_video.jsonl
"""
import os

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
os.environ.setdefault('FPS', '3')
os.environ.setdefault('DURATION', '5.0')

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path

import pandas as pd
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = 'You are an expert video analyst.\nPlease think about the question as if you were a human pondering deeply. It’s encouraged to include self-reflection or verification in the reasoning process. Then, give a final verdict within <answer> </answer> tags.'
USER_PROMPT = 'Is this video real or fake?'
FPS = int(os.environ['FPS'])


def load_vision_process(repo : Path):
	# importa só o arquivo, sem puxar o pacote swift inteiro
	spec = importlib.util.spec_from_file_location('vision_process', repo / 'swift' / 'qwen_vl_utils' / 'vision_process.py')
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def parse_answer(text : str):
	match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.IGNORECASE | re.DOTALL)
	if not match:
		return None
	answer = match.group(1).lower()
	if ('real' in answer) != ('fake' in answer):
		return 'real' if 'real' in answer else 'fake'
	return None


def answer_prob(completion) -> float | None:
	if not completion.logprobs:
		return None
	text = ''
	for token_id, step in zip(completion.token_ids, completion.logprobs):
		if text.rstrip().endswith('<answer>'):
			p = { 'real': 0.0, 'fake': 0.0 }
			for cand in step.values():
				word = (cand.decoded_token or '').strip().lower()
				if word in p:
					p[word] += math.exp(cand.logprob)
			if p['real'] + p['fake'] > 0:
				return p['fake'] / (p['real'] + p['fake'])
			return None
		text += step[token_id].decoded_token or ''
	return None


def build_input(vision_process, processor, path : str) -> dict:
	message =\
	[
		{ 'role': 'system', 'content': SYSTEM_PROMPT },
		{ 'role': 'user', 'content': [ { 'type': 'video', 'video': path, 'fps': FPS }, { 'type': 'text', 'text': USER_PROMPT } ] }
	]
	text = processor.apply_chat_template(message, tokenize = False, add_generation_prompt = True)
	_, videos, video_kwargs = vision_process.process_vision_info(message, image_patch_size = processor.image_processor.patch_size, return_video_kwargs = True, return_video_metadata = True)
	video_kwargs['do_resize'] = False
	return { 'prompt': text, 'multi_modal_data': { 'video': videos }, 'mm_processor_kwargs': video_kwargs }


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--clips', required = True)
	parser.add_argument('--out', required = True)
	parser.add_argument('--model', required = True)
	# pasta que contém swift/qwen_vl_utils/vision_process.py (padrão: cópia em third_party/; ou um clone do repositório oficial)
	parser.add_argument('--repo', default = str(Path(__file__).resolve().parents[1] / 'third_party' / 'VideoVeritas'))
	parser.add_argument('--batch', type = int, default = 32)
	parser.add_argument('--limit', type = int, default = 0)
	args = parser.parse_args()

	out = Path(args.out)
	done = { json.loads(line)['path'] for line in out.read_text().splitlines() } if out.exists() else set()
	clips = pd.read_csv(args.clips)
	items = [ r._asdict() for r in clips.itertuples(index = False) if r.path not in done ]
	items = items[:args.limit] if args.limit else items
	print(f'{len(items)} clipes a processar ({len(done)} já feitos)', flush = True)

	vision_process = load_vision_process(Path(args.repo))
	processor = AutoProcessor.from_pretrained(args.model)
	llm = LLM(model = args.model, max_model_len = int(os.environ.get("MAX_LEN", "20480")), gpu_memory_utilization = float(os.environ.get("GPU_UTIL", "0.92")), limit_mm_per_prompt = { 'image': 0, 'video': 1 }, max_num_seqs = 16)
	sampling = SamplingParams(temperature = 0, max_tokens = 4096, logprobs = 5)

	with out.open('a') as out_file:
		for start in range(0, len(items), args.batch):
			batch = items[start:start + args.batch]
			outputs = llm.generate([ build_input(vision_process, processor, item['path']) for item in batch ], sampling, use_tqdm = False)
			for item, output in zip(batch, outputs):
				completion = output.outputs[0]
				row = { **item, 'pred': parse_answer(completion.text), 'p_fake': answer_prob(completion), 'text': completion.text, 'finish': completion.finish_reason, 'n_prompt_tokens': len(output.prompt_token_ids), 'n_tokens': len(completion.token_ids) }
				out_file.write(json.dumps(row, ensure_ascii = False) + '\n')
			out_file.flush()
			print(f'{min(start + args.batch, len(items))}/{len(items)}', flush = True)


if __name__ == '__main__':
	main()
