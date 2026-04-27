# Docker Runtime Config

`make_container.sh` requires this directory.

## Tokens

Create one local-only token file before starting the container:

```bash
cp docker/config/.tokens.example docker/config/.tokens
vim docker/config/.tokens
chmod 600 docker/config/.tokens
```

`.tokens` supports:

```bash
GITHUB_TOKEN=...
HF_TOKEN=...
WANDB_API_KEY=...
```

The real `.tokens` file is intentionally ignored by git.

## Volumes And Ports

Container runtime settings, volumes, and ports are managed in `docker/config/runtime.env`.

```bash
IMAGE_NAME=jaehee-base:0404
CONTAINER_NAME=jaehee-contaccum-refine
WORKSPACE_DIR=/workspace
VOLUMES="/home/jaeheekim/codes:/workspace /media/data:/data"
EXTRA_VOLUMES=
PORTS=9204:9204
```

Use whitespace or commas for multiple entries, e.g.:

```bash
PORTS="9204:9204 7860:7860"
EXTRA_VOLUMES="/tmp:/tmp,/scratch:/scratch"
```
