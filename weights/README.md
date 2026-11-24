# Kronos-mini weights (transport branch)

Official `NeoQuasar/Kronos-mini` and `NeoQuasar/Kronos-Tokenizer-2k` weights (MIT), stored here only so
cloud sessions without Hugging Face access can load them.

| File | SHA256 | Matches recorded official hash |
| --- | --- | --- |
| `Kronos-mini/model.safetensors` | `a7d5f37e2e9fbd9891f7d7d4f72574512dd1f704fee14223e0a8cd0fbf54197c` | yes |
| `Kronos-Tokenizer-2k/model.safetensors` | `b97ec46b3b72160509e289183eaf7bdf5f0dac5bb9b49522f6d46638a99a8717` | yes |

The two `config.json` files were **reconstructed from the tensor shapes**, not downloaded, so their
hashes differ from the official ones. Architecture fields load with `strict=True`; dropout and
tokenizer-loss fields (`*_dropout_p`, `beta`, `gamma*`, `zeta`) are inactive in `eval()` and are
placeholders. Replace them with the official files before any fine-tuning.
