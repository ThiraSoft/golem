"""Times the checkpoint's own system_one on the requests laya/bench_test.go
times, so that the README's comparison is the same work on the same machine.

Usage (from a venv that holds torch, transformers and safetensors):
    python ref/laya/bench.py <checkpoint_dir> [cpu|cuda]

On cuda it runs as system_one does there, under bf16 autocast.
"""

import sys
import time

import torch

STATE = {"subject": "Refund not received",
         "body": "I cancelled my subscription two weeks ago and I still have not seen the refund on my card. "
                 "Order #88213. Can someone please look into this? Thanks, Maria"}
QS = {
    "department": {"type": "choice", "instructions": "Which team should handle this email?",
                   "criteria": {"billing": "payments, refunds, invoices", "support": "technical problems",
                                "sales": "pricing and new purchases"}},
    "urgency": {"type": "score", "instructions": "How urgent is this message?",
                "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"]},
    "is_refund": {"type": "noul", "instructions": "The customer is asking about a refund."},
    "tone": {"type": "choice", "instructions": "What is the tone of the writer?",
             "criteria": ["angry", "neutral", "polite", "sarcastic", "desperate", "happy"]},
}


def main() -> None:
    ckpt, device = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "cpu")
    sys.path.insert(0, ckpt)
    from rl_agent_api import RLAgent  # noqa: E402

    agent = RLAgent(ckpt, device=device)
    many = {f"{k}{'_' * i}": v for i in range(8) for k, v in QS.items()}
    for name, qs in (("1q", {"is_refund": QS["is_refund"]}), ("4q", QS), ("32q", many)):
        for _ in range(3):
            agent.system_one(STATE, qs)
        runs = 10 if device != "cpu" else 3
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            out = agent.system_one(STATE, qs)
        if device != "cpu":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / runs
        print(f"{name}: {dt * 1000:.1f} ms, {out['usage']['input_tokens'] / dt:.0f} tok/s")


if __name__ == "__main__":
    main()
