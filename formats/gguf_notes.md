# GGUF notes (why it is not the Phase-1 format)

- GGUF is a single-file container with a key/value header and block-quantized
  tensor data (Q4_K, Q6_K, ...). `mx.load` can read GGUF, but MLX materializes
  dequantized or mlx-quantized forms depending on type support, and per-tensor
  lazy semantics are less predictable than safetensors.
- Quantized blocks are laid out for llama.cpp's kernels. To use them with
  `mx.quantized_matmul` they must be repacked (group size / scale layout differ) —
  a conversion step that duplicates the model on disk, which this project avoids.
- llama.cpp itself already does mmap-based lazy paging of GGUF (the OS pages
  weights in on demand) — but it offers no *policy* control (pinning, budgets,
  quantize-on-load, KV spill), which is the point of this runtime.
- Our quantize-on-load path (`runtime/quant.py`) gets the main benefit associated
  with GGUF (small resident footprint) while keeping one full-precision artifact
  on disk and per-module precision policy at runtime.

Revisit GGUF when: we want pre-quantized *disk* reads (streamed 4-bit reads would
cut the per-token disk bill 4× for the pure-streaming regime — the "70B on 16GB"
scenario), at which point a GGUF→safetensors-q repack tool or a native GGUF block
reader becomes worth building.
