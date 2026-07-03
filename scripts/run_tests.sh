#!/bin/bash
# Media Agent 测试运行脚本
# 支持不同级别的测试和完整测试套件

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
# 测试级别定义
# ============================================================================

# Level 0: 快速单元测试（不需要外部依赖）
run_unit_tests() {
    echo_header "Level 0: 单元测试"
    
    # 运行 lib 单元测试
    cargo test --lib --no-fail-fast 2>&1 | tee test_results_unit.log
    
    if [ $? -eq 0 ]; then
        echo_success "单元测试通过"
        return 0
    else
        echo_error "单元测试失败"
        return 1
    fi
}

# Level 1: 工作流和执行引擎测试
run_workflow_tests() {
    echo_header "Level 1: 工作流测试"
    
    cargo test --test workflow_test --no-fail-fast 2>&1 | tee test_results_workflow.log
    
    if [ $? -eq 0 ]; then
        echo_success "工作流测试通过"
        return 0
    else
        echo_error "工作流测试失败"
        return 1
    fi
}

run_execution_tests() {
    echo_header "Level 1: 执行引擎测试"
    
    cargo test --test execution_test --no-fail-fast 2>&1 | tee test_results_execution.log
    
    if [ $? -eq 0 ]; then
        echo_success "执行引擎测试通过"
        return 0
    else
        echo_error "执行引擎测试失败"
        return 1
    fi
}

# Level 2: Agent 单元测试（不需要 LLM 后端）
run_agent_unit_tests() {
    echo_header "Level 2: Agent 单元测试"
    
    cargo test --test agent_unit_test --no-fail-fast 2>&1 | tee test_results_agent_unit.log
    
    if [ $? -eq 0 ]; then
        echo_success "Agent 单元测试通过"
        return 0
    else
        echo_error "Agent 单元测试失败"
        return 1
    fi
}

# Level 3: 集成测试（需要 stable-diffusion.cpp，可选）
run_integration_tests() {
    echo_header "Level 3: 集成测试"
    
    # 检查后端是否可用
    if ! check_backend_available; then
        echo_warning "后端不可用，跳过集成测试"
        echo_warning "请先启动 stable-diffusion.cpp 后端"
        return 0
    fi
    
    cargo test --test integration_test --no-fail-fast 2>&1 | tee test_results_integration.log
    
    if [ $? -eq 0 ]; then
        echo_success "集成测试通过"
        return 0
    else
        echo_error "集成测试失败"
        return 1
    fi
}

# Level 4: Agent E2E 测试（需要 llama.cpp，大部分 marked as #[ignore]）
run_agent_e2e_tests() {
    echo_header "Level 4: Agent E2E 测试"
    
    # 检查 LLM 后端是否可用
    if ! check_llm_backend_available; then
        echo_warning "LLM 后端不可用，跳过 E2E 测试"
        echo_warning "请先启动 llama.cpp server: ./scripts/start_llama_server.sh"
        return 0
    fi
    
    # 运行非 ignored 测试
    cargo test --test agent_e2e_test --no-fail-fast 2>&1 | tee test_results_agent_e2e_basic.log
    
    # 可选：运行 ignored 测试（需要完整后端）
    echo_warning "以下测试需要完整后端支持，已标记为 #[ignore]:"
    echo "  - test_agent_chat_endpoint"
    echo "  - test_agent_full_generation"
    echo "  - test_pdca_*"
    echo "  - test_dag_*"
    echo ""
    echo "运行完整 E2E 测试请使用: ./scripts/run_tests.sh --full-e2e"
    
    return 0
}

# Level 5: 完整 E2E 测试（运行所有 #[ignore] 测试）
run_full_e2e_tests() {
    echo_header "Level 5: 完整 E2E 测试"
    
    # 启动所有必要后端
    echo "启动后端服务..."
    
    # 启动 llama.cpp
    if ! pgrep -f "llama-server" > /dev/null; then
        echo "启动 llama.cpp server..."
        ./scripts/start_llama_server.sh &
        sleep 5
    fi
    
    # 启动 stable-diffusion.cpp（如果有）
    if [ -f "./scripts/start_sd_cpp_server.sh" ]; then
        if ! pgrep -f "sd-cpp" > /dev/null; then
            echo "启动 stable-diffusion.cpp server..."
            ./scripts/start_sd_cpp_server.sh &
            sleep 5
        fi
    fi
    
    # 启动 media_agent server
    echo "启动 media_agent server..."
    cargo run --bin comfyui-server &
    MEDIA_AGENT_PID=$!
    sleep 3
    
    # 运行所有测试（包括 ignored）
    cargo test --test agent_e2e_test -- --ignored --no-fail-fast 2>&1 | tee test_results_full_e2e.log
    
    # 清理
    echo "清理进程..."
    kill $MEDIA_AGENT_PID 2>/dev/null || true
    
    if [ $? -eq 0 ]; then
        echo_success "完整 E2E 测试通过"
        return 0
    else
        echo_error "完整 E2E 测试失败"
        return 1
    fi
}

# ============================================================================
# 辅助函数
# ============================================================================

check_backend_available() {
    # 检查 stable-diffusion.cpp 后端
    curl -s http://localhost:8188/health > /dev/null 2>&1
    return $?
}

check_llm_backend_available() {
    # 检查 llama.cpp server
    curl -s http://localhost:8081/health > /dev/null 2>&1
    return $?
}

check_server_running() {
    # 检查 media_agent server
    curl -s http://localhost:8188/health > /dev/null 2>&1
    return $?
}

# ============================================================================
# 主测试流程
# ============================================================================

run_all_tests() {
    echo_header "Media Agent 全量测试套件"
    
    FAILED=0
    PASSED=0
    
    # Level 0
    if run_unit_tests; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
    
    # Level 1
    if run_workflow_tests; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
    
    if run_execution_tests; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
    
    # Level 2
    if run_agent_unit_tests; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
    
    # Level 3（可选）
    run_integration_tests
    # 不计入失败（可选测试）
    
    # Level 4（可选）
    run_agent_e2e_tests
    # 不计入失败（可选测试）
    
    echo_header "测试结果汇总"
    echo -e "${GREEN}通过: $PASSED${NC}"
    echo -e "${RED}失败: $FAILED${NC}"
    
    if [ $FAILED -eq 0 ]; then
        echo_success "所有核心测试通过！"
        return 0
    else
        echo_error "有测试失败，请检查日志"
        return 1
    fi
}

# ============================================================================
# 快速测试（仅核心单元测试）
# ============================================================================

run_quick_tests() {
    echo_header "快速测试（核心单元测试）"
    
    cargo test --lib -- --test-threads=4 2>&1 | tail -20
    
    if [ $? -eq 0 ]; then
        echo_success "快速测试通过"
        return 0
    else
        echo_error "快速测试失败"
        return 1
    fi
}

# ============================================================================
# 测试覆盖率
# ============================================================================

run_coverage() {
    echo_header "测试覆盖率分析"
    
    # 检查是否有 cargo-tarpaulin
    if ! command -v cargo-tarpaulin &> /dev/null; then
        echo_warning "cargo-tarpaulin 未安装"
        echo "安装: cargo install cargo-tarpaulin"
        return 1
    fi
    
    cargo tarpaulin --out Html --out Stdout 2>&1 | tee coverage_report.log
    
    echo_success "覆盖率报告已生成: tarpaulin-report.html"
}

# ============================================================================
# 命令行参数处理
# ============================================================================

case "$1" in
    --unit)
        run_unit_tests
        ;;
    --workflow)
        run_workflow_tests
        ;;
    --execution)
        run_execution_tests
        ;;
    --agent-unit)
        run_agent_unit_tests
        ;;
    --integration)
        run_integration_tests
        ;;
    --agent-e2e)
        run_agent_e2e_tests
        ;;
    --full-e2e)
        run_full_e2e_tests
        ;;
    --quick)
        run_quick_tests
        ;;
    --coverage)
        run_coverage
        ;;
    --all)
        run_all_tests
        ;;
    --help)
        echo "Media Agent 测试运行脚本"
        echo ""
        echo "用法: $0 [选项]"
        echo ""
        echo "选项:"
        echo "  --unit          运行单元测试（Level 0）"
        echo "  --workflow      运行工作流测试（Level 1）"
        echo "  --execution     运行执行引擎测试（Level 1）"
        echo "  --agent-unit    运行 Agent 单元测试（Level 2）"
        echo "  --integration   运行集成测试（Level 3，需要后端）"
        echo "  --agent-e2e     运行 Agent E2E 基础测试（Level 4，需要 LLM）"
        echo "  --full-e2e      运行完整 E2E 测试（Level 5，需要全部后端）"
        echo "  --quick         快速测试（仅核心单元测试）"
        echo "  --coverage      生成测试覆盖率报告"
        echo "  --all           运行全量测试套件"
        echo "  --help          显示帮助信息"
        echo ""
        echo "示例:"
        echo "  $0 --quick      # 快速验证核心功能"
        echo "  $0 --all        # 完整测试（不包括 #[ignore] 测试）"
        echo "  $0 --full-e2e   # 包括 #[ignore] 的完整 E2E 测试"
        ;;
    *)
        # 默认运行全量测试
        run_all_tests
        ;;
esac