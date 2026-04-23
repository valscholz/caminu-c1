#!/usr/bin/env bash
# Smoke-test the local llama-server with three kinds of prompts:
#   1. plain text
#   2. vision (attach a JPEG)
#   3. tool calling (require a tool_call back)
# Run this AFTER ./run.sh is up (llama-server on 127.0.0.1:8080).
set -euo pipefail

URL="${LLAMA_URL:-http://127.0.0.1:8080}"

echo "=== 1. Text turn ==="
curl -s "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages":[
      {"role":"system","content":"You are Caminu C1, a concise voice assistant. Keep replies under 15 words."},
      {"role":"user","content":"Say hi to Val."}
    ],
    "max_tokens": 60,
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'

echo ""
echo "=== 2. Vision turn (OAK-D snapshot) ==="
IMG_PATH="${1:-/tmp/oakd_snap3.jpg}"
[ -f "$IMG_PATH" ] || { echo "no image at $IMG_PATH — skipping vision test"; exit 0; }
B64=$(base64 -w0 "$IMG_PATH")
curl -s "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"messages\":[
      {\"role\":\"system\",\"content\":\"You are Caminu C1. Keep replies under 30 words.\"},
      {\"role\":\"user\",\"content\":[
        {\"type\":\"text\",\"text\":\"What do you see in this image? Be concise.\"},
        {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,${B64}\"}}
      ]}
    ],
    \"max_tokens\": 80,
    \"temperature\": 0.5
  }" | jq -r '.choices[0].message.content'

echo ""
echo "=== 3. Tool-calling turn ==="
curl -s "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages":[
      {"role":"system","content":"You are a helpful assistant. Use tools when they apply."},
      {"role":"user","content":"What time is it right now?"}
    ],
    "tools":[
      {"type":"function","function":{
        "name":"get_time",
        "description":"Return the current local time as ISO-8601.",
        "parameters":{"type":"object","properties":{},"required":[]}
      }}
    ],
    "max_tokens": 80,
    "temperature": 0.5
  }' | jq '.choices[0].message'
