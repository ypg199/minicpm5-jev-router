---
base_model: openbmb/MiniCPM5-2B
library_name: peft
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-classification
tags:
- lora
- peft
- tool-routing
- function-calling
- decision-model
- jev-alternative
- minicpm5
datasets:
- Salesforce/xlam-function-calling-60k
- ZefanCai/Open-Jev
- tatsu-lab/alpaca
---

# MiniCPM5-2B Jev Router (LoRA)

A LoRA adapter that turns [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) into a small Jev-style decision model: given a state and a question with fixed options, it answers with one option letter and a calibrated probability for every option, in a single forward pass, with no text generation.

It was trained mainly for **tool routing**: given the available tools and a user message, should the assistant call a tool, ask for missing information, say it can't help, or answer directly, and which tool to call.

Code, notebooks and a Jev-compatible server: https://github.com/<GITHUB_REPO>

## Results

All rows use the same questions. "Zero-shot" rows were scored by me with the same letter-probability method; Kev and Jev numbers are the ones published in [Kev's repository](https://github.com/jaredpalmer/kev) (fp32, their own decision head).

| Model | Params | BFCL routing (1,140 q) | When2Call dev / test | Kev transfer-v4 dev / test |
|---|---|---|---|---|
| **MiniCPM5-2B + this LoRA** | 2.5B | **0.811** | **0.687 / 0.707** | **0.712 / 0.739** |
| MiniCPM5-2B, zero-shot | 2.5B | 0.769 | 0.360 / 0.373 | 0.657 / 0.654 |
| Qwen3.5-4B, zero-shot | 4B | 0.845 | 0.547 / 0.527 | 0.712 / 0.745 |
| Qwen3.5-9B, zero-shot | 9B | 0.887 | 0.613 / 0.500 | 0.787 / 0.802 |
| Kev-0.8B | 0.8B | – | 0.167 / 0.133 | 0.648 / 0.697 |
| Kev-4B | 4B | – | 0.660 / – | 0.817 / 0.838 |
| Kev-9B | 9B | – | – / 0.680 | 0.820 / 0.852 |
| Jev (TypeSafe, hosted) | – | – | 0.760 / – | 0.857 / – |

- **When2Call** (call / ask for info / can't help / answer directly, built on BFCL): the adapter roughly doubles the base model and is level with Kev-4B and Kev-9B. With 150 questions per split, differences under about 8 points are within noise.
- **transfer-v4** (Kev's out-of-domain decisions): +8.5 points over the base model and ahead of Kev-0.8B, still about 10 points behind Kev-4B.
- **BFCL** (pick the right tool or NONE): better than the base model, still behind zero-shot Qwen3.5-4B and 9B.
- Validation set (same generator as training, 1,202 rows): accuracy 0.956, calibration error 0.018 (0.012 after the stored temperature).

Nothing from BFCL, When2Call or Kev's suites was used for training, and training rows matching any evaluation item were removed. When2Call is built from BFCL-style tools and xLAM is similar in style, so it is not a fully out-of-distribution test.

## How to use

The model answers with the next-token probability of the option letters after a fixed prompt. The prompts must match training exactly, so use the helpers in the GitHub repo (`router/prompts.py`) or the server:

```bash
git clone https://github.com/<GITHUB_REPO> && cd minicpm5-jev-router
pip install -r requirements.txt
ROUTER_ADAPTER=<HF_REPO> uvicorn router.server:app --port 8009
```

```bash
curl -s localhost:8009/v1/action -H 'content-type: application/json' -d '{
  "user_message": "Show my statement.",
  "tools": [{"name": "get_card_statement", "description": "Fetch the statement for a card and month.",
             "parameters": {"type": "dict", "required": ["card_id", "month"],
                            "properties": {"card_id": {"type": "string"}, "month": {"type": "string"}}}}]}'
# {"action": "request_for_info", "confidence": ..., "probabilities": {...}}
```

`POST /v1/systemone` accepts the TypeSafe/Jev request shape (choice, yes/no and score questions), so Kev's benchmark can score it with `--remote`.

## Training

- Data: 14,789 rows. About 11,400 routing decisions generated from xLAM-60k (call / ask: a required detail deleted / decline: the right tool removed, near-miss tools kept / answer directly: Alpaca instructions with unrelated tools), plus about 3,400 general decisions from Open-Jev (game and control sources excluded).
- LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections (25.1M trainable parameters, 0.99%).
- Loss on the single answer token only. 1 epoch, 393 steps, lr 2e-4 cosine, about 16k tokens per step; 126 minutes on one Kaggle T4 in fp16.
- Temperature fitted on the validation set: 0.916 (stored in `calibration.json`).

## Limitations

- Options are letters A–Z, so at most 25 tools plus NONE per question.
- Weak at arithmetic-style decisions (date differences) and knowledge-heavy questions, like other models of this size.
- Kev/Jev comparisons are against their published numbers (fp32, their own decision head); this model was scored in fp16 with letter probabilities, so the comparison is close but not fully controlled.
- English only in training.

## License

The adapter is released under **CC BY-NC 4.0** because part of the training data (Alpaca instructions) is CC BY-NC 4.0. The base model is Apache-2.0, xLAM-60k is CC BY 4.0 (Salesforce) and Open-Jev is CC0. The code in the GitHub repo is Apache-2.0.
