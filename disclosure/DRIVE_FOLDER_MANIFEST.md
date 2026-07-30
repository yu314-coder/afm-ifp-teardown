# Drive folder — what to upload

Assembled by `disclosure/build_drive_folder.sh` into `/Volumes/D/github/afm-ifp-teardown/local/apple_disclosure/`.

## Include

| Item | What it is |
|---|---|
| `afm_teardown.pdf` | The technical report, 23 pages. Original analysis only. |
| `src/` | The decoder and analysis source. Original code. |
| `*.md` | Per-topic result write-ups referenced by the report. |
| `*.json` | Derived maps and inventories — offsets, shapes, block classes. Measurements, not content. |

## Exclude — deliberately

| Item | Why |
|---|---|
| `*.gguf` | Contains Apple's weights and tokenizer. |
| `*.pt`, `*.npz`, `*.npy` | Decoded weight tensors. |
| `tok_vocab.json` | Apple's tokenizer vocabulary. |
| `*.dmg`, `*.hwx`, `*.odix`, `*.asset` | Apple's shipped assets. |
| Process cores | Contain live model data. |

The report discloses the *format* — offsets, codec, permutation, closed forms, validation
statistics. That is what makes the disclosure actionable. The extracted content adds nothing to
it and changes the character of the act from analysis to redistribution.

## Sharing setting

Set to **Restricted**, then share with the specific address that replies. Not "anyone with the
link" — Drive links forward, and a forwarded link holding another company's model weights is a
publication.
