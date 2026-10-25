# ZeroGPU compatibility assessment (work in progress)

This branch is an experimental deployment track. The existing full local implementation is preserved at `local-full`; do not merge this branch to `main` until the acceptance checks below pass.

## Confirmed current architecture
- UI: Gradio.
- Speech-to-text: `sherpa-onnx` NVIDIA Parakeet TDT 0.6B v3 **ONNX CPU provider**, cached lazily.
- Semantic embeddings: SentenceTransformer instantiated with `device="cpu"`, cached lazily.
- Melodic features: NumPy/PyAV CPU processing.
- Catalog: Gradio `State` with a mutable FAISS-backed index and in-memory feature arrays.
- Import/compare handlers share a serial queue.

## ZeroGPU constraints
- Current ZeroGPU requires Gradio plus at least one genuine `@spaces.GPU` callback.
- The decorator invokes GPU work in a separate worker; in-memory state and non-trivial runtime objects must not be assumed to share identity across its process boundary.
- Torch models intended for GPU execution should be instantiated/placed on CUDA at module scope, per Hugging Face's official ZeroGPU guidance.
- Current ONNX ASR is CPU-only: wrapping it in `@spaces.GPU` does **not** accelerate it and wastes visitor GPU quota.
- Python versions documented for ZeroGPU are 3.10.13 and 3.12.12 (as of 2026-10-25); source code targets >=3.11, so prefer Python 3.12.12 for ZeroGPU.

## Implementation checkpoints
1. Prototype a **genuine** CUDA-enabled embedding backend in isolation, using a compatible Torch model and `@spaces.GPU` for that computation only. Do not move the FAISS catalog into the GPU worker.
2. Keep the embedding output a plain CPU NumPy array. Apply catalog mutations on the main Gradio process.
3. Decide separately whether to preserve CPU-only Parakeet under ZeroGPU (no GPU quota benefit, model-download and memory limits) or build a tested GPU-capable ASR backend **without misrepresenting the model**.
4. Validate upload, YouTube search, lyrics fallback, metadata-only entries, embedding retrieval, two-song comparisons, session isolation and error paths.
5. Verify rebuild and real external-service functionality on the target ZeroGPU hardware, including quota/time limits.
6. Only merge into `main` after a deployed end-to-end demonstration works. Keep `local-full` unchanged.

## Deployment safety
Do not add a no-op GPU decorator simply to make startup pass. Do not assume the build or app runtime is successful until observed via logs and user-facing tests. Never introduce paid inference credentials or hardware upgrades.

References:
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://huggingface.co/buckets/huggingface/skills/tree/skills/huggingface-zerogpu/references/how-zerogpu-works.md
