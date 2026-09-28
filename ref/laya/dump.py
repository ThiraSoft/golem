"""Records what PyTorch computes for Laya, the decision model Convai published
as convaiinnovations/laya, so that the Go tests can hold golem's pass to it.

Usage (from a venv that holds torch, transformers and safetensors):
    python ref/laya/dump.py <checkpoint_dir> <out_dir>

<checkpoint_dir> is the Hugging Face repository as it downloads: model.safetensors,
rl_agent_config.json, rl_common.py, encoder/ and tokenizer/. The model is run on
the processor in float32, from the fp16 weights the file holds, which is what
golem's processor path computes too.

Writes, under <out_dir>:

  tokenizer.json   texts and the identifiers Hugging Face's tokenizer gives them
  cases.json       per case: the question, the sequence build_sequence made, the
                   marker positions, the calibrated answer system_one returns
  <case>/*.bin     float32 waypoints of that case, row-major, one row a position

Nothing this writes is versioned: see the .gitignore.
"""

import json
import sys
from pathlib import Path

import torch

STATE = {
    "subject": "Refund not received",
    "body": "I cancelled my subscription two weeks ago and I still have not seen the refund "
            "on my card. Order #88213. Can someone please look into this? Thanks, Maria",
}

# A state longer than the sliding window of 128, so that the local layers and
# the global ones see different things.
LONG_STATE = {
    "ticket": "Our production cluster went down at 03:12 UTC. " * 12
              + "The on-call engineer restarted the ingress controller, which brought "
                "traffic back, but latency is still twice the usual. We need to know "
                "whether to page the database team or wait for the morning shift.",
    "severity_reported": "high",
    "customer": {"tier": "enterprise", "seats": 1200},
}

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this email?",
                   "criteria": {"billing": "payments, refunds, invoices", "support": "technical problems",
                                "sales": "pricing and new purchases"}},
    "urgency": {"type": "score", "instructions": "How urgent is this message?",
                "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"]},
    "is_refund": {"type": "noul", "instructions": "The customer is asking about a refund."},
    "tone": {"type": "choice", "instructions": "What is the tone of the writer?",
             "criteria": ["angry", "neutral", "polite", "sarcastic", "desperate", "happy"]},
}

LONG_QUESTIONS = {
    "escalate": {"type": "noul", "instructions": "The database team should be paged now.",
                 "criteria": {"true": "page them", "false": "wait for the morning"}},
    "impact": {"type": "score", "instructions": "How large is the business impact?",
               "criteria": ["none", "minor", "major", "severe", "total outage"]},
}

# French, for the multilingual checkpoint, and for the others to be held to
# the same text.
FR_STATE = {
    "de": "jeanne.martin@exemple.fr",
    "objet": "Prélèvement en double",
    "message": "Bonjour, j'ai été prélevée deux fois pour la facture de septembre (n° 2026-0913). "
               "Pourriez-vous me rembourser rapidement ? C'est la deuxième fois cette année… Merci d'avance.",
}

FR_QUESTIONS = {
    "service": {"type": "choice", "instructions": "Quel service doit traiter ce courriel ?",
                "criteria": {"facturation": "paiements, remboursements, factures", "support": "problèmes techniques",
                             "commercial": "tarifs et nouveaux achats"}},
    "urgence": {"type": "score", "instructions": "Quelle est l'urgence de ce message ?",
                "criteria": ["pas urgent", "peu urgent", "urgent", "critique"]},
    "remboursement": {"type": "noul", "instructions": "La cliente demande un remboursement.",
                      "criteria": {"true": "oui", "false": "non"}},
}

TOKENIZER_TEXTS = [
    "Bonjour, j'ai été prélevée deux fois : pourriez-vous m'aider ?",
    "L'œuvre « Les Misérables » coûte 12,50 € – c'est donné !",
    "  deux espaces devant, et   trois au milieu",
    "Ça, c'était l'été où ÉLODIE a dit « non ».",
    "ligne une\nligne deux\n\n\tindentée",
    "Hello world",
    " leading space",
    "a   b",
    "trailing   ",
    "it's we're they'll I'd",
    "numbers 1234567 and 3.14159",
    "tabs\tand\nnewlines\n\n",
    "Ünïcödé café naïve 日本語 😀",
    '{"subject": "Refund not received", "n": 42}',
    "choice question: Which team should handle this email?",
    "   indented by three",
    "punctuation!!! ??? ... ---",
    "code: for (int i = 0; i < n; ++i) { x += y; }",
    "a" + " " * 30 + "b",
    "URLs https://example.com/path?q=1&r=2",
]


def main() -> None:
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ckpt))
    from rl_agent_api import RLAgent  # noqa: E402
    from rl_common import QTYPES, build_sequence, render_options  # noqa: E402

    torch.manual_seed(0)
    agent = RLAgent(str(ckpt), device="cpu")
    model, tok, cfg = agent.model, agent.tok, agent.cfg
    model.float().eval()

    (out / "tokenizer.json").write_text(json.dumps(
        [{"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]} for t in TOKENIZER_TEXTS],
        ensure_ascii=False, indent=1))

    cases = []
    for name, state, questions in (("short", STATE, QUESTIONS), ("long", LONG_STATE, LONG_QUESTIONS),
                                   ("fr", FR_STATE, FR_QUESTIONS)):
        answers = agent.system_one(state, questions)["answers"]
        for qid, qdef in questions.items():
            q = agent._to_internal(qdef)
            ids, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
            case = f"{name}_{qid}"
            record(model, ids, markers, QTYPES[q["t"]], out / case)
            cases.append({"name": case, "state": state, "question": qdef, "ids": ids,
                          "markers": markers, "answer": answers[qid]})
            print(case, len(ids), "tokens", len(markers), "options")
    (out / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1))


@torch.no_grad()
def record(model, ids, markers, qtype, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    got = {}

    def keep(name):
        def hook(_mod, _inp, output):
            if isinstance(output, tuple):
                output = output[0]
            got[name] = output.detach().float()[0].contiguous()
        return hook

    enc = model.encoder
    handles = [enc.embeddings.register_forward_hook(keep("embed"))]
    for i, layer in enumerate(enc.layers):
        handles.append(layer.register_forward_hook(keep(f"layer-{i}")))
    handles.append(enc.final_norm.register_forward_hook(keep("encoded")))
    for i, layer in enumerate(model.head.layers):
        handles.append(layer.register_forward_hook(keep(f"head-{i}")))

    input_ids = torch.tensor([ids])
    att = torch.ones_like(input_ids)
    mpos = torch.tensor([markers])
    mmask = torch.ones_like(mpos, dtype=torch.bool)
    logits, act = model(input_ids, att, mpos, mmask, torch.tensor([qtype]))
    for h in handles:
        h.remove()
    got["logits"] = logits[0].float()
    got["act"] = act[0].float()

    shapes = {}
    for name, t in got.items():
        (out / f"{name}.bin").write_bytes(t.numpy().tobytes())
        shapes[name] = list(t.shape)
    (out / "index.json").write_text(json.dumps(shapes, indent=1))


if __name__ == "__main__":
    main()
