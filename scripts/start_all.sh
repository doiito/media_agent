#!/bin/bash
# media_agent 服务器启动脚本
# 包含 llama.cpp server 和 media_agent 的完整启动

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== media_agent + gliding_horse 启动脚本 ===${NC}"
echo ""

# 1. 检查配置文件
CONFIG_FILE="${CONFIG_FILE:-config/agent.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}Warning: Config file not found: $CONFIG_FILE${NC}"
    echo "Using default configuration..."
fi

# 2. 创建必要目录
mkdir -p data/agent_memory output logs skills workflows

# 3. 启动 llama.cpp server（后台）
echo -e "${GREEN}Starting llama.cpp server (LLM backend)...${NC}"
if [ -f scripts/start_llama_server.sh ]; then
    # 检查 llama.cpp server 是否已运行
    if pgrep -f "llama-server" > /dev/null; then
        echo -e "${YELLOW}llama.cpp server already running, skipping...${NC}"
    else
        # 启动 llama.cpp server（后台）
        LLAMA_PORT="${LLAMA_PORT:-8081}"
        ./scripts/start_llama_server.sh &
        LLAMA_PID=$!
        echo "llama.cpp server started (PID: $LLAMA_PID, port: $LLAMA_PORT)"
        
        # 等待 server 就绪
        echo "Waiting for llama.cpp server to be ready..."
        for i in {1..30}; do
            if curl -s http://localhost:$LLAMA_PORT/health > /dev/null 2>&1; then
                echo -e "${GREEN}llama.cpp server is ready!${NC}"
                break
            fi
            sleep 1
        done
        
        if ! curl -s http://localhost:$LLAMA_PORT/health > /dev/null 2>&1; then
            echo -e "${RED}Error: llama.cpp server failed to start${NC}"
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}Warning: llama.cpp server script not found${NC}"
    echo "Please ensure LLM backend is available at http://localhost:8081/v1"
fi

# 4. 启动 media_agent server
echo -e "${GREEN}Starting media_agent server...${NC}"
SERVER_PORT="${SERVER_PORT:-8188}"

# 使用 cargo run 启动（开发模式）
# 或者使用编译后的二进制文件
if [ -f target/release/comfyui-server ]; then
    ./target/release/comfyui-server --config "$CONFIG_FILE"
elif [ -f target/debug/comfyui-server ]; then
    ./target/debug/comfyui-server --config "$CONFIG_FILE"
else
    echo "Building and running..."
    cargo run --bin comfyui-server -- --config "$CONFIG_FILE"
fi

# 说明：
# 端点列表：
#   - http://localhost:8188/health         : 健康检查
#   - http://localhost:8188/agent/chat     : Agent 对话（自然语言）
#   - http://localhost:8188/agent/status   : Agent 状态
#   - http://localhost:8188/agent/init     : 初始化 Agent
#   - http://localhost:8188/agent/workflows: 工作流列表
#   - http://localhost:8188/ws             : WebSocket 连接
#
# 测试命令：
#   curl -X POST http://localhost:8188/agent/chat \
#     -H "Content-Type: application/json" \
#     -d '{"message":"画一只赛博朋克风格的猫"}'