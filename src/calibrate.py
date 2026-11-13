import json
from dataclasses import asdict, replace
from math import ceil
import numpy as np
from mocker import Mocker, MockerLatency, Row, soldr_ttft


def calibrate_ab(p: MockerLatency) -> tuple[float, float]:
    chunk = p.chunk
    rows: list[Row] = []
    next_hash = 0
    for isl in [512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 245760]:
        for frac in [0, 0.25, 0.5, 0.75, 0.95]:
            hit, hashes = int(isl * frac) // 512 * 512, list(range(next_hash, next_hash + ceil(isl / 512)))
            next_hash += len(hashes)
            if hit:
                rows.append({"timestamp": 60_000 * len(rows), "input_length": hit, "output_length": 1,
                             "hash_ids": hashes[:hit // 512]})
            rows.append({"timestamp": 60_000 * len(rows), "input_length": isl, "output_length": 1, "hash_ids": hashes})
    xs, ys = [], []
    for r, q in zip(rows, Mocker(p, 512).run(rows, 1)):
        new, hit = r["input_length"] - q["reused_input_tokens"], q["reused_input_tokens"]
        xs.append([new, (r["input_length"] ** 2 - hit ** 2) / 2, ceil(new / chunk)])
        ys.append(q["ttft_ms"])
    x, y = np.array(xs), np.array(ys)
    (a, b, _), *_ = np.linalg.lstsq(x / y[:, None], np.ones(len(y)), rcond=None)
    return float(a), float(b)


def calibrate(path: str, workers: int, config: str) -> None:
    p = MockerLatency.load(config)
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    recs = Mocker(p, 512).run(rows, workers)
    target = sum(q["ttft_ms"] for q in recs) / len(recs)
    decode_ms = round(sum(q["last_token_ms"] - q["first_token_ms"] for q in recs)
                      / sum(q["output_length"] - 1 for q in recs))
    a, b = calibrate_ab(p)
    p = replace(p, decode_ms=decode_ms, a=a, b=b)
    # Bisect for c
    lo, hi = 0.0, 100.0
    for _ in range(20):
        p = replace(p, c=(lo + hi) / 2)
        if soldr_ttft(rows, recs, p) * decode_ms < target:
            lo = p.c
        else:
            hi = p.c
    p = replace(p, c=round((lo + hi) / 2, 1))
    with open(f"params_{config}.json", "w") as f:
        json.dump(asdict(p), f, indent=1)
    print(f"{p} (Mocker mean TTFT {target:.1f} ms)")
