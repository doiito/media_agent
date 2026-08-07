#!/bin/bash
# llama.cpp server 启动脚本（OpenAI 兼容模式）
# 用于 Agent LLM 网关

# 配置参数
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/dev-data/ai-test/llama.cpp-b9810}"
CONFIG_FILE="${CONFIG_FILE:-config/config.json}"
CONFIGURED_MODEL=""
if [ -f "$CONFIG_FILE" ] && command -v jq >/dev/null 2>&1; then
    CONFIGURED_MODEL="$(jq -r '.llama_cpp.model_path // ""' "$CONFIG_FILE")"
fi
MODEL_PATH="${LLAMA_CPP_MODEL_PATH:-${MODEL_PATH:-${CONFIGURED_MODEL:-models/qwen2.5-7b-instruct-q4_k_m.gguf}}}"
PORT="${LLAMA_PORT:-8081}"
HOST="${LLAMA_HOST:-127.0.0.1}"
CTX_SIZE="${LLAMA_CTX_SIZE:-32768}"
PARALLEL="${LLAMA_PARALLEL:-1}"
THREADS="${LLAMA_THREADS:-8}"
GPU_LAYERS="${LLAMA_GPU_LAYERS:-auto}"
FIT_TARGET_MIB="${LLAMA_FIT_TARGET_MIB:-1024}"
SLEEP_IDLE_SECONDS="${LLAMA_SLEEP_IDLE_SECONDS:-1}"
MODEL_ALIAS="${AGENT_LLM_MODEL:-local-model}"

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo "Please download a model first, e.g.:"
    echo "Set LLAMA_CPP_MODEL_PATH to an instruct GGUF or configure llama_cpp.model_path."
    exit 1
fi

# 检查 llama-server 是否存在
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_DIR/build/bin/llama-server}"
if [ ! -x "$LLAMA_SERVER" ]; then
    # 尝试在常见位置查找
    for path in "$LLAMA_CPP_DIR/build/bin/llama-server" "/usr/local/bin/llama-server"; do
        if [ -x "$path" ]; then
            LLAMA_SERVER="$path"
            break
        fi
    done
    
    if [ ! -x "$LLAMA_SERVER" ]; then
        echo "Error: llama-server not found"
        echo "Please build llama.cpp with server support:"
        echo "  cd llama.cpp && make llama-server"
        exit 1
    fi
fi

echo "Starting llama.cpp server..."
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"
echo "  Host: $HOST"
echo "  Context size: $CTX_SIZE"
echo "  Parallel slots: $PARALLEL"
echo "  Threads: $THREADS"
echo "  GPU layers: $GPU_LAYERS"
echo "  GPU fit reserve: ${FIT_TARGET_MIB} MiB"
echo "  Idle sleep: ${SLEEP_IDLE_SECONDS}s"
echo "  API model alias: $MODEL_ALIAS"
echo ""
echo "OpenAI compatible endpoint: http://localhost:$PORT/v1"
echo ""

# 启动 llama-server
$LLAMA_SERVER \
    -m "$MODEL_PATH" \
    --port "$PORT" \
    --host "$HOST" \
    --ctx-size "$CTX_SIZE" \
    --parallel "$PARALLEL" \
    --threads "$THREADS" \
    --n-gpu-layers "$GPU_LAYERS" \
    --fit on \
    --fit-target "$FIT_TARGET_MIB" \
    --sleep-idle-seconds "$SLEEP_IDLE_SECONDS" \
    --flash-attn on \
    --alias "$MODEL_ALIAS" \
    --metrics \
    --log-disable

# 说明：
# --metrics: 启用 Prometheus 指标端点 (/metrics)
# --log-disable: 禁用默认日志输出（减少噪音）
#
# 更多参数选项：
#   --gpu-layers N    : 使用 GPU 加速（N 层）
#   --batch-size N    : 批处理大小
#   --temp N          : 温度参数
#   --top-k N         : Top-K 采样
#   --top-p N         : Top-P 采样
