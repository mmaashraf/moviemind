#!/usr/bin/env bash
set -euo pipefail

# MovieMind helper: install/check/start Ollama and pull selected model.
# Usage:
#   bash scripts/setup_local_ollama.sh
#   OLLAMA_MODEL=llama3.1:8b bash scripts/setup_local_ollama.sh

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

echo "[MovieMind] Local Ollama setup starting..."
echo "[MovieMind] target model: ${OLLAMA_MODEL}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "[MovieMind] Ollama CLI not found."
  if command -v brew >/dev/null 2>&1; then
    echo "[MovieMind] Installing Ollama via Homebrew..."
    brew install ollama
  else
    echo "[MovieMind] Homebrew not found. Install Ollama manually from https://ollama.com/download"
    exit 1
  fi
fi

if ! curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "[MovieMind] Ollama service not reachable at ${OLLAMA_URL}. Starting service..."
  if command -v brew >/dev/null 2>&1; then
    brew services start ollama >/dev/null 2>&1 || true
  fi
  # If brew service did not start or is unavailable, start background process.
  if ! curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    nohup ollama serve >/tmp/moviemind_ollama.log 2>&1 &
    sleep 2
  fi
fi

if ! curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "[MovieMind] Ollama is still unreachable at ${OLLAMA_URL}."
  echo "[MovieMind] Please run 'ollama serve' manually and retry."
  exit 1
fi

echo "[MovieMind] Pulling model ${OLLAMA_MODEL} (if missing)..."
ollama pull "${OLLAMA_MODEL}"

echo "[MovieMind] Local Ollama setup complete."
echo "[MovieMind] Quick test:"
echo "curl -X POST ${OLLAMA_URL}/api/generate -d '{\"model\":\"${OLLAMA_MODEL}\",\"prompt\":\"hello\",\"stream\":false}'"
