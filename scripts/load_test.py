"""R5: classifier p99 over HTTP with 16 concurrent clients. Start `make serve` first."""
import asyncio
import json
import sys
import time

import httpx
import numpy as np

from shifa import config as C
from shifa.data import load_split

URL = "http://127.0.0.1:8000"
CONC, TOTAL = 16, 800


async def worker(client, texts, q, lat):
    while True:
        try:
            i = q.get_nowait()
        except asyncio.QueueEmpty:
            return
        t0 = time.perf_counter()
        r = await client.post(f"{URL}/v1/classify", json={"text": texts[i % len(texts)]})
        r.raise_for_status()
        lat.append((time.perf_counter() - t0) * 1000)


async def main():
    texts = list(load_split("val").text)
    async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=CONC)) as c:
        health = (await c.get(f"{URL}/v1/health")).json()
        for t in texts[:20]:                                   # warm up
            await c.post(f"{URL}/v1/classify", json={"text": t})
        q, lat = asyncio.Queue(), []
        for i in range(TOTAL):
            q.put_nowait(i)
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(c, texts, q, lat) for _ in range(CONC)])
        wall = time.perf_counter() - t0
    lat = np.array(lat)
    res = {"model": health["topic_model"], "concurrency": CONC, "requests": TOTAL, "p50_ms": float(np.percentile(lat, 50)),
           "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99)), "rps": TOTAL / wall,
           "budget_ms": 40, "within_budget": bool(np.percentile(lat, 99) <= 40)}
    C.REP.mkdir(exist_ok=True)
    (C.REP / "http_load.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["within_budget"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
