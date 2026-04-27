# Docker Runtime Config

`make_container.sh` requires this directory and mounts it into the container.

Create these local-only files before starting the container:

```bash
mkdir -p docker/config/gh docker/config/github docker/config/huggingface
cp docker/config/github/token.example docker/config/github/token
cp docker/config/huggingface/token.example docker/config/huggingface/token
vim docker/config/github/token
vim docker/config/huggingface/token
chmod 600 docker/config/github/token docker/config/huggingface/token
```

The token files are intentionally ignored by git.
