set shell := ["bash", "-uc"]

default:
    @just --list

install:
    uv sync --extra dev

# Build all datasets used by AttachIQ:
#   * 7,200-prompt text dataset (10 request classes)
#   * 8,000-image RVL-CDIP subset (8 document classes)
#   * 10,000-row standard fusion features (26-dim)
#   * 2,360-row hand-labelled hard rubric fusion dataset (26-dim)
data:
    uv run python -m attachiq.data.build_text_dataset
    uv run python -m attachiq.data.build_image_dataset
    uv run python -m attachiq.data.build_fusion_dataset
    uv run python -m attachiq.data.build_hard_fusion_dataset

train: train-image train-text train-fusion

train-text:
    uv run python -m attachiq.training.train_text

train-image:
    uv run python -m attachiq.training.train_image --arch deep --epochs 20

train-fusion:
    uv run python -m attachiq.training.train_fusion --source union

evaluate:
    uv run python -m attachiq.evaluation.evaluate

evaluate-image:
    uv run python -m attachiq.evaluation.evaluate_image

demo:
    uv run streamlit run src/attachiq/ui/streamlit_app.py

cli *ARGS:
    uv run python -m attachiq.inference.cli {{ARGS}}

test:
    uv run pytest

clean:
    rm -rf .venv .pytest_cache .ruff_cache __pycache__ build dist
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type d -name .pytest_cache -exec rm -rf {} +

# Stage all, commit when there are changes, then push (one recipe).
#   just push
#   just push "Your commit message"
push *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    git add -A
    msg="${*:-chore: update}"
    if git diff --cached --quiet; then
      echo "Nothing to commit (clean index after add)."
    else
      git commit -m "$msg"
    fi
    git push

