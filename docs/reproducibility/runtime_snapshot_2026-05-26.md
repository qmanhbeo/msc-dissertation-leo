# Runtime Reproducibility Snapshot (2026-05-26)

- Repository: `dissertation`
- Git branch: `main`
- Git commit: `0c3295c`
- Snapshot UTC: `2026-05-26T05:51:26Z`
- Snapshot local (Europe/London): `2026-05-26T06:51:26+0100`

## OS / Kernel / WSL

- Distro: `Ubuntu 24.04.3 LTS (Noble Numbat)`
- Kernel: `Linux 5.15.146.1-microsoft-standard-WSL2`
- Platform string: `Linux-5.15.146.1-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- `/proc/version`: `Linux version 5.15.146.1-microsoft-standard-WSL2 ... #1 SMP Thu Jan 11 04:09:03 UTC 2024`

## GPU / Driver

From `nvidia-smi` during active embedding run:

- NVIDIA driver version: `546.30`
- NVIDIA-SMI version: `545.35`
- Reported CUDA version (driver): `12.3`
- GPU: `NVIDIA GeForce RTX 3050 Laptop GPU`
- VRAM: `4096 MiB`

## Conda / Python Environment

- Conda version: `26.1.1`
- Runtime env used for this job: `/home/manh/miniforge3/envs/dissertation`
- Python executable: `/home/manh/miniforge3/envs/dissertation/bin/python`
- Python version: `3.11.15` (conda-forge build)
- pip version: `26.0.1`

## PyTorch / CUDA Runtime

Pinned for this run:

- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `torchaudio==2.5.1+cu121`

Observed values in elevated run context (same context used to run the job):

- `torch.__version__ = 2.5.1+cu121`
- `torch.version.cuda = 12.1`
- `torch.cuda.is_available() = True`
- CUDA device count: `1`
- CUDA device 0: `NVIDIA GeForce RTX 3050 Laptop GPU`

Note: in non-elevated sandbox probes, `torch.cuda.is_available()` may appear `False`; the actual job context is elevated and GPU-enabled.

## Key Library Versions Used

- `sentence-transformers==5.5.1`
- `transformers==4.57.6`
- `tokenizers==0.22.2`
- `numpy==2.4.4`
- `scipy==1.17.1`
- `scikit-learn==1.8.0`
- `pandas==3.0.3`
- `datasets==2.19.1`

## Reproducibility Artifacts Generated

- Python lockfile: `docs/reproducibility/requirements.lock.txt`
- Conda explicit spec: `docs/reproducibility/conda-explicit-dissertation.txt`
- Conda env YAML (no-builds): `docs/reproducibility/conda-env-dissertation.yml`

## Job Command (resume-safe)

```bash
/home/manh/miniforge3/envs/dissertation/bin/python -u code/run_full_corpus_pipeline.py --device cuda --batch-size 256 --local-files-only
```

Current pipeline structure remains checkpoint-safe via manifests and stage status under:

- `data/1_preprocessed/research_corpus/metadata/manifest.json`
- `data/2_embedded/research_shards/metadata/manifest.json`
- `data/2_embedded/research_shards/metadata/*.json`
