#!/bin/bash
# llama.cpp server 启动脚本（OpenAI 兼容模式）
# 用于 Agent LLM 网关

# 配置参数
MODEL_PATH="${MODEL_PATH:-models/qwen2.5-7b-instruct-q4_k_m.gguf}"
PORT="${LLAMA_PORT:-8081}"
HOST="${LLAMA_HOST:-0.0.0.0}"
CTX_SIZE="${CTX_SIZE:-4096}"
THREADS="${THREADS:-4}"

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo "Please download a model first, e.g.:"
    echo "  wget -O models/qwen2.5-7b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
    exit 1
fi

# 检查 llama-server 是否存在
LLAMA_SERVER="${LLAMA_SERVER:-./llama-server}"
if [ ! -f "$LLAMA_SERVER" ]; then
    # 尝试在常见位置查找
    for path in "./llama-server" "./llama.cpp/llama-server" "../llama.cpp/llama-server"; do
        if [ -f "$path" ]; then
            LLAMA_SERVER="$path"
            break
        fi
    done
    
    if [ ! -f "$LLAMA_SERVER" ]; then
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
echo "  Threads: $THREADS"
echo ""
echo "OpenAI compatible endpoint: http://localhost:$PORT/v1"
echo ""

# 启动 llama-server
$LLAMA_SERVER \
    -m "$MODEL_PATH" \
    --port "$PORT" \
    --host "$HOST" \
    --ctx-size "$CTX_SIZE" \
    --threads "$THREADS" \
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