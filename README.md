# MiniCPM5 Jev Router

A 2.5B open Jev-style decision model for tool routing. A LoRA adapter on [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B), trained for about two hours on one free Kaggle T4.

Like TypeSafe's [Jev](https://docs.typesafe.ai), it doesn't generate text. It reads the state and a question with fixed options, and returns a probability for every option in one forward pass. Its main job: given the available tools and a user message, decide whether the assistant should **call a tool, ask for missing information, say it can't help, or answer directly**, and which tool to call.

## Results

| Model | Params | BFCL routing (1,140 q) | When2Call dev / test | Kev transfer-v4 dev / test |
|---|---|---|---|---|
| **MiniCPM5-2B + this LoRA** | 2.5B | **0.811** | **0.687 / 0.707** | **0.712 / 0.739** |
| MiniCPM5-2B, zero-shot | 2.5B | 0.769 | 0.360 / 0.373 | 0.657 / 0.654 |
| Qwen3.5-4B, zero-shot | 4B | 0.845 | 0.547 / 0.527 | 0.712 / 0.745 |
| Qwen3.5-9B, zero-shot | 9B | 0.887 | 0.613 / 0.500 | 0.787 / 0.802 |
| Kev-0.8B | 0.8B | – | 0.167 / 0.133 | 0.648 / 0.697 |
| Kev-4B | 4B | – | 0.660 / – | 0.817 / 0.838 |
| Kev-9B | 9B | – | – / 0.680 | 0.820 / 0.852 |
| Jev (hosted) | – | – | 0.760 / – | 0.857 / – |

- **Tool routing (When2Call):** the LoRA roughly doubles the base model (0.373 → 0.707) and lands level with [Kev](https://github.com/jaredpalmer/kev)'s fine-tuned 4B and 9B models at 2.5B parameters. Each split has 150 questions, so gaps under about 8 points are noise.
- **General decisions (Kev's transfer-v4):** +8.5 points over the base model and ahead of Kev-0.8B; still about 10 points behind Kev-4B.
- **Picking the exact tool (BFCL):** better than the base model, still behind zero-shot Qwen3.5-4B/9B. That's the target for the next version.

Kev and Jev numbers are the ones Kev publishes (fp32, their own decision head). Everything else was scored here with the same questions and the same letter-probability method in fp16. When2Call and transfer-v4 use Kev's frozen files and Kev's own metric code. All numbers are in [`results/scorecard.csv`](results/scorecard.csv).

## Run it

```bash
pip install -r requirements.txt
ROUTER_ADAPTER=<adapter on the Hub or a local folder> uvicorn router.server:app --port 8009
python examples/client.py
```

| Endpoint | In | Out |
|---|---|---|
| `POST /v1/action` | `user_message`, `tools` | `tool_call` / `request_for_info` / `cannot_answer` / `direct` with probabilities |
| `POST /v1/route` | `messages` or `user_message`, `tools` | the tool to call, or `null` for none, with probabilities |
| `POST /v1/systemone` | TypeSafe/Jev request shape (`choice`, `noul`, `score` questions) | Jev-shaped answers |

Because `/v1/systemone` speaks Jev's format, Kev's benchmark can score this model directly:

```bash
python -m kev.benchmark --remote http://127.0.0.1:8009 --suite evals/v4/transfer-v4 --out runs/router
```

## How it was built

Everything runs on Kaggle's free GPUs. The notebooks are in [`notebooks/`](notebooks):

1. **`01_compare_models.ipynb`** scores models as decision models: tools as lettered options plus NONE, one forward pass, probabilities read from the letter logits. It covers BFCL, Kev's transfer-v4 and When2Call, and a native tool-calling check, and it can load a LoRA adapter.
2. **`02_build_dataset.ipynb`** builds 16k training rows. xLAM-60k becomes call / ask (a required detail deleted) / decline (the right tool removed, near-miss tools kept) / direct (Alpaca instructions) examples, and Open-Jev adds general decisions. Anything matching an evaluation item is removed.
3. **`03_train_lora.ipynb`** trains a LoRA (r=16, all projections) with the loss on the answer letter only, then fits a temperature for calibration.
4. **`04_publish.ipynb`** uploads the adapter to Hugging Face and smoke-tests the server.

What the zero-shot comparison showed before training: MiniCPM5 already picked the right tool as well as Qwen3.5-9B (93.5% on BFCL when a tool fits). But it almost never declined or asked for missing details, because it matched tools by topic. The training data was weighted towards exactly those cases.

## Limitations

- At most 25 tools plus NONE per question (letters A–Z).
- English only. Weak at date arithmetic and knowledge-heavy questions.
- Its own validation set (0.956) comes from the same generator as the training data. Trust the benchmark table above, not that number.

## License and credits

The code is Apache-2.0. The adapter is CC BY-NC 4.0, because the training data includes Alpaca instructions (CC BY-NC 4.0).

Built on [MiniCPM5](https://github.com/OpenBMB/MiniCPM) (OpenBMB), [xLAM-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (Salesforce, CC BY 4.0), [Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) (CC0), [BFCL](https://github.com/ShishirPatil/gorilla), [When2Call](https://huggingface.co/datasets/nvidia/When2Call) and [Kev](https://github.com/jaredpalmer/kev), whose evaluation suites and metric code are used here. `router/prompts.py` reuses two small helpers from Kev (Apache-2.0). Not affiliated with TypeSafe.
