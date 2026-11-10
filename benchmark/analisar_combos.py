"""Compara as configurações do VideoVeritas (combos/*.jsonl) com a base R1 (14 frames, 768 tokens, raciocínio).

Por vídeo: score = média de P(fake) nos clipes usados; veredito = regra do protocolo
(raciocínio: >= 2 de 3 clipes 'fake'; direto: média de P(fake) > 0,5).
Variantes calculadas a posteriori (sem rodar de novo): parada antecipada (exata para a regra 2 de 3),
só o clipe do meio (c1) e só 2 clipes (c0+c1).
Tempo de CPU ESTIMADO a partir do benchmark medido (26 frames/768 tokens no i9-14900K, fp32, 24 threads):
  leitura da entrada ≈ 190 s / 9538 tokens ≈ 0,0199 s por token de prompt (linear: conservador p/ prompts menores);
  geração ≈ 1,0 s por token de saída.
Uso: python benchmark/analisar_combos.py
  VV_COMBOS: pasta com os <nome>.jsonl de gpu/combos_gpu.py (padrão: benchmark/combos)
  VV_ORIG:   (opcional) predições de gpu/05_inferir_video.py, para conferir se a base R1 reproduz a rodada original
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "R1_14f_768t_raciocinio"
CPU_PREFILL_S_POR_TOKEN = 190.0 / 9538
CPU_DECODE_S_POR_TOKEN = 1.0
SEED = 11


def load(path):
    d = pd.read_json(path, lines=True)
    d["y"] = (d.label == "fake").astype(int)
    return d


def per_video(d, clips=(0, 1, 2), regra="auto"):
    d = d[d["clip"].isin(clips)].copy()
    modo = d.modo.iloc[0] if "modo" in d else "raciocinio"
    # clipes sem P(fake) parseável: usa o veredito textual (1/0); conta quantos
    d["p"] = d.p_fake.where(d.p_fake.notna(), (d.pred == "fake").astype(float))
    d["voto"] = (d.pred == "fake").astype(float) if modo == "raciocinio" else (d.p > 0.5).astype(float)
    v = d.groupby("video_id").agg(y=("y", "first"), p=("p", "mean"), votos=("voto", "mean"), n=("clip", "size"))
    if regra == "auto":
        regra = "votos" if modo == "raciocinio" else "media"
    v["fake_pred"] = (v.votos > 0.5) if regra == "votos" else (v.p > 0.5)
    return v


def boot(y, s, base_s=None, n=2000):
    rng = np.random.default_rng(SEED)
    r, f = np.where(y == 0)[0], np.where(y == 1)[0]
    a, dd = [], []
    for _ in range(n):
        i = np.r_[rng.choice(r, r.size), rng.choice(f, f.size)]
        x = roc_auc_score(y[i], s[i]); a.append(x)
        if base_s is not None:
            dd.append(x - roc_auc_score(y[i], base_s[i]))
    ci = np.percentile(a, [2.5, 97.5])
    return ci, (np.percentile(dd, [2.5, 97.5]) if dd else None)


def row(nome, d, v, base_v, clips_medios=3.0):
    v = v.loc[base_v.index]
    y, s = v.y.values, v.p.values
    ci, dci = boot(y, s, base_v.p.values)
    auc = roc_auc_score(y, s)
    rr, ff = (~v.fake_pred[v.y == 0]).mean(), v.fake_pred[v.y == 1].mean()
    used = d[d["clip"].isin(range(int(np.ceil(clips_medios))))] if clips_medios < 3 else d
    n_prompt, n_out = d.n_prompt.mean(), d.n_out.mean()
    cpu_clip = n_prompt * CPU_PREFILL_S_POR_TOKEN + n_out * CPU_DECODE_S_POR_TOKEN
    gpu_clip = d.t_infer_lote.mean() + d.t_preproc_lote.mean() if "t_infer_lote" in d else np.nan
    return {"configuração": nome, "AUC": round(auc, 3), "IC95": f"[{ci[0]:.3f}–{ci[1]:.3f}]",
            "ΔAUC vs base [IC95]": f"{auc - roc_auc_score(base_v.y, base_v.p):+.3f} [{dci[0]:+.3f}; {dci[1]:+.3f}]",
            "reais certos": f"{rr:.0%} ({rr * 35:.0f}/35)", "fakes certos": f"{ff:.0%} ({ff * 95:.0f}/95)",
            "acc bal.": round((rr + ff) / 2, 3), "mesmo veredito da base": f"{(v.fake_pred == base_v.fake_pred).mean():.0%}",
            "clipes/vídeo": round(clips_medios, 2), "tokens entrada/clipe": int(round(n_prompt)), "tokens saída/clipe": int(round(n_out)),
            "GPU s/vídeo (lote)": round(gpu_clip * clips_medios, 1), "CPU min/vídeo (estim.)": round(cpu_clip * clips_medios / 60, 1)}


def main():
    files = sorted(glob.glob(os.path.join(os.environ.get("VV_COMBOS", os.path.join(HERE, "combos")), "*.jsonl")))
    D = {os.path.basename(f)[:-6]: load(f) for f in files}
    if BASE not in D:
        raise SystemExit("base R1 ainda não rodou")
    base_d = D[BASE]
    base_v = per_video(base_d)

    # reprodução da base contra a rodada original (fps=3, que dá 14 frames)
    orig_path = os.environ.get("VV_ORIG")
    orig = load(orig_path) if orig_path and os.path.exists(orig_path) else None
    if orig is not None:
        m = base_d.merge(orig, on=["video_id", "clip"], suffixes=("", "_orig"))
        print(f"[reprodução] R1 × rodada original: {len(m)} clipes; veredito igual em {(m.pred == m.pred_orig).mean():.1%}; "
              f"|ΔP(fake)| mediana {np.nanmedian((m.p_fake - m.p_fake_orig).abs()):.2e}")
        ov = per_video(orig.assign(modo="raciocinio"))
        print(f"[reprodução] AUC por vídeo: original {roc_auc_score(ov.y, ov.p):.3f} × R1 {roc_auc_score(base_v.y, base_v.p):.3f}")
    for k, d in D.items():
        bad = int(d.p_fake.isna().sum()); trunc = int((d.finish == "length").sum()) if d.modo.iloc[0] == "raciocinio" else 0
        if bad or trunc:
            print(f"[aviso] {k}: {bad} clipes sem P(fake) parseável, {trunc} respostas truncadas")

    rows = []
    for k, d in D.items():
        rows.append(row(k, d, per_video(d), base_v))
    # variantes a posteriori
    for k in [x for x in D if x.endswith("raciocinio")]:
        d = D[k]
        w = d.pivot_table(index="video_id", columns="clip", values="pred", aggfunc="first")
        concord = (w[0] == w[1]).mean()
        v = per_video(d)  # parada antecipada dá EXATAMENTE o mesmo veredito da regra 2 de 3
        rows.append(row(f"{k} + parada antecipada", d, v, base_v, clips_medios=2 * concord + 3 * (1 - concord)))
    for k in [BASE, "R5_14f_768t_direto"]:
        if k in D:
            d = D[k]
            rows.append(row(f"{k} só clipe do meio", d[d["clip"] == 1], per_video(d, (1,), "media"), base_v, 1))
            rows.append(row(f"{k} só 2 clipes (c0+c1)", d[d["clip"].isin([0, 1])], per_video(d, (0, 1), "media"), base_v, 2))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(HERE, "combos_resultado.csv"), index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
