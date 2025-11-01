import argparse, time, csv, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def run_bench(model_name, batch, new_tokens, fusion, csv_out):
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={'': device})

    prompt = "Explain rotary position embeddings briefly."
    inputs = tokenizer([prompt]*batch, return_tensors='pt', padding=True).to(device)

    # Warmup prefill
    _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    iters = 3
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times)/len(times)
    toks = batch * new_tokens
    tps = toks / avg
    print(f"[{model_name}] fusion={fusion} batch={batch} new={new_tokens} -> {tps:.2f} tok/s")

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    write_header = not os.path.exists(csv_out)
    with open(csv_out, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['model','fusion','batch','new_tokens','tokens_per_sec','avg_seconds'])
        w.writerow([model_name, fusion, batch, new_tokens, f"{tps:.2f}", f"{avg:.4f}"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--new_tokens', type=int, default=128)
    ap.add_argument('--fusion', choices=['on','off'], default='off')
    ap.add_argument('--csv', default='results/bench.csv')
    args = ap.parse_args()
    run_bench(args.model, args.batch, args.new_tokens, args.fusion, args.csv)
