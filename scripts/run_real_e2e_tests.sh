#!/bin/bash
# DeepSeek API 真实 E2E 测试启动脚本
# 使用真实 LLM 模型进行测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

echo_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

echo_error() {
    echo -e "${RED}✗ $1${NC}"
}

echo_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ============================================================================
# 环境变量检查
# ============================================================================

check_env_vars() {
    echo_header "检查环境变量"
    
    # DEEPSEEK_API_URL
    if [ -z "$DEEPSEEK_API_URL" ]; then
        echo_warning "DEEPSEEK_API_URL 未设置，使用默认值"
        export DEEPSEEK_API_URL="https://api.deepseek.com/v1"
        echo "  设置为: $DEEPSEEK_API_URL"
    else
        echo_success "DEEPSEEK_API_URL = $DEEPSEEK_API_URL"
    fi
    
    # DEEPSEEK_API_KEY
    if [ -z "$DEEPSEEK_API_KEY" ]; then
        echo_error "DEEPSEEK_API_KEY 未设置！"
        echo ""
        echo "请设置 API Key："
        echo "  export DEEPSEEK_API_KEY=your_api_key"
        echo ""
        echo "获取 API Key："
        echo "  1. 访问 https://platform.deepseek.com"
        echo "  2. 注册并登录"
        echo "  3. 在 API Keys 页面创建新 Key"
        exit 1
    else
        echo_success "DEEPSEEK_API_KEY 已设置（长度：${#DEEPSEEK_API_KEY}）"
    fi
    
    # 可选：DEEPSEEK_MODEL
    if [ -z "$DEEPSEEK_MODEL" ]; then
        export DEEPSEEK_MODEL="deepseek-chat"
        echo "  使用默认模型: $DEEPSEEK_MODEL"
    else
        echo_success "DEEPSEEK_MODEL = $DEEPSEEK_MODEL"
    fi
}

# ============================================================================
# API 连接测试
# ============================================================================

test_api_connection() {
    echo_header "测试 DeepSeek API 连接"
    
    # 发送简单测试请求
    echo "发送测试请求..."
    
    RESPONSE=$(curl -s -X POST "$DEEPSEEK_API_URL/chat/completions" \
        -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello, reply OK"}],
            "max_tokens": 50
        }')
    
    if [ $? -eq 0 ]; then
        # 检查响应是否包含有效内容
        if echo "$RESPONSE" | grep -q "choices"; then
            echo_success "API 连接成功"
            echo "响应: $(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null || echo "$RESPONSE")"
            return 0
        else
            echo_error "API 响应异常"
            echo "响应: $RESPONSE"
            return 1
        fi
    else
        echo_error "API 连接失败"
        return 1
    fi
}

# ============================================================================
# 运行真实 E2E 测试
# ============================================================================

run_real_e2e_tests() {
    echo_header "运行 DeepSeek 真实 E2E 测试"
    
    # 编译项目（如果需要）
    echo "编译项目..."
    cargo build 2>&1 | tail -5
    
    # 运行真实 E2E 测试
    echo ""
    echo "运行 agent_real_e2e_test..."
    cargo test --test agent_real_e2e_test --no-fail-fast 2>&1 | tee test_results_real_e2e.log
    
    # 检查结果
    if grep -q "test result: ok" test_results_real_e2e.log; then
        echo_success "真实 E2E 测试通过"
        
        # 显示统计
        PASSED=$(grep -c "test ... ok" test_results_real_e2e.log || echo 0)
        FAILED=$(grep -c "test ... FAILED" test_results_real_e2e.log || echo 0)
        
        echo ""
        echo "测试统计："
        echo "  通过: $PASSED"
        echo "  失败: $FAILED"
    else
        echo_error "真实 E2E 测试有失败"
        
        # 显示失败详情
        grep "FAILED" test_results_real_e2e.log || true
    fi
}

# ============================================================================
# 运行特定测试
# ============================================================================

run_specific_test() {
    TEST_NAME=$1
    
    echo_header "运行测试: $TEST_NAME"
    
    cargo test --test agent_real_e2e_test "$TEST_NAME" --no-fail-fast -- --nocapture 2>&1
}

# ============================================================================
# 完整测试流程
# ============================================================================

run_full_test_suite() {
    echo_header "DeepSeek 真实模型完整测试流程"
    
    # 1. 检查环境变量
    check_env_vars
    
    # 2. 测试 API 连接
    test_api_connection || {
        echo_error "API 连接失败，请检查网络和 API Key"
        exit 1
    }
    
    # 3. 运行真实 E2E 测试
    run_real_e2e_tests
    
    echo_header "测试完成"
}

# ============================================================================
# 单个测试（用于调试）
# ============================================================================

case "$1" in
    --check-env)
        check_env_vars
        ;;
    --test-connection)
        check_env_vars
        test_api_connection
        ;;
    --connection)
        test_api_connection
        ;;
    --all)
        run_full_test_suite
        ;;
    --e2e)
        check_env_vars
        run_real_e2e_tests
        ;;
    --test)
        if [ -z "$2" ]; then
            echo "用法: $0 --test <测试名称>"
            echo ""
            echo "可用测试："
            echo "  test_deepseek_api_connection"
            echo "  test_deepseek_chinese_chat"
            echo "  test_deepseek_tool_calling"
            echo "  test_agent_pdca_intent_parse"
            echo "  test_agent_param_optimization"
            echo "  test_agent_react_loop"
            echo "  test_agent_multi_turn_memory"
            echo "  test_batch_generate_intent"
            echo "  test_workflow_builder_with_llm"
            echo "  test_system_prompt_effects"
            echo "  test_deepseek_response_time"
            echo "  test_concurrent_api_calls"
            exit 1
        fi
        check_env_vars
        run_specific_test "$2"
        ;;
    --quick)
        check_env_vars
        echo "运行快速测试..."
        cargo test --test agent_real_e2e_test test_deepseek_api_connection --no-fail-fast -- --nocapture
        cargo test --test agent_real_e2e_test test_deepseek_chinese_chat --no-fail-fast -- --nocapture
        ;;
    --help)
        echo "DeepSeek 真实 E2E 测试脚本"
        echo ""
        echo "用法: $0 [选项]"
        echo ""
        echo "选项:"
        echo "  --check-env        检查环境变量设置"
        echo "  --test-connection  测试 API 连接"
        echo "  --connection       快速 API 连接测试"
        echo "  --all              运行完整测试流程"
        echo "  --e2e              仅运行 E2E 测试"
        echo "  --test <名称>      运行特定测试"
        echo "  --quick            快速验证（2个基础测试）"
        echo "  --help             显示帮助"
        echo ""
        echo "环境变量（必须）："
        echo "  DEEPSEEK_API_KEY   DeepSeek API Key"
        echo ""
        echo "环境变量（可选）："
        echo "  DEEPSEEK_API_URL   API URL（默认：https://api.deepseek.com/v1）"
        echo "  DEEPSEEK_MODEL     模型名（默认：deepseek-chat）"
        echo ""
        echo "示例："
        echo "  export DEEPSEEK_API_KEY=sk-xxx"
        echo "  $0 --all"
        ;;
    *)
        run_full_test_suite
        ;;
esac