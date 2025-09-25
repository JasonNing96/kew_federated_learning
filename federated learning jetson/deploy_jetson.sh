#!/bin/bash
# Jetson Nano 集群部署脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   FLQ 联邦学习 Jetson 集群部署${NC}"
echo -e "${BLUE}========================================${NC}"

# 配置参数
MASTER_IP="192.168.1.100"  # 修改为你的 master 节点 IP
WORKER_IPS=("192.168.1.101" "192.168.1.102" "192.168.1.103" "192.168.1.104")  # 修改为你的 worker 节点 IP
USERNAME="jetson"  # Jetson Nano 默认用户名
PROJECT_DIR="/home/$USERNAME/flq_federated_learning"

# SSH 密钥文件（如果使用密钥认证）
SSH_KEY=""  # 如: ~/.ssh/id_rsa

echo -e "${YELLOW}部署配置:${NC}"
echo -e "  Master IP: $MASTER_IP"
echo -e "  Worker IPs: ${WORKER_IPS[*]}"
echo -e "  用户名: $USERNAME"
echo -e "  项目目录: $PROJECT_DIR"

# SSH 命令构建
build_ssh_cmd() {
    local ip=$1
    local cmd="ssh"
    if [[ -n "$SSH_KEY" ]]; then
        cmd="$cmd -i $SSH_KEY"
    fi
    cmd="$cmd $USERNAME@$ip"
    echo "$cmd"
}

# SCP 命令构建
build_scp_cmd() {
    local src=$1
    local ip=$2
    local dst=$3
    local cmd="scp"
    if [[ -n "$SSH_KEY" ]]; then
        cmd="$cmd -i $SSH_KEY"
    fi
    cmd="$cmd $src $USERNAME@$ip:$dst"
    echo "$cmd"
}

# 检查连接
check_connection() {
    local ip=$1
    echo -e "${BLUE}检查连接: $ip${NC}"
    if $(build_ssh_cmd $ip) "echo 'Connected'" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ $ip 连接成功${NC}"
        return 0
    else
        echo -e "${RED}✗ $ip 连接失败${NC}"
        return 1
    fi
}

# 部署文件到节点
deploy_files() {
    local ip=$1
    local node_type=$2
    
    echo -e "${BLUE}部署文件到 $ip ($node_type)${NC}"
    
    # 创建项目目录
    $(build_ssh_cmd $ip) "mkdir -p $PROJECT_DIR"
    
    # 复制核心文件
    $(build_scp_cmd "flq_${node_type}.py" $ip "$PROJECT_DIR/")
    $(build_scp_cmd "flq_config.json" $ip "$PROJECT_DIR/")
    $(build_scp_cmd "start_${node_type}.sh" $ip "$PROJECT_DIR/")
    $(build_scp_cmd "environment.yml" $ip "$PROJECT_DIR/")
    
    echo -e "${GREEN}✓ 文件部署完成: $ip${NC}"
}

# 设置 conda 环境
setup_conda_env() {
    local ip=$1
    echo -e "${BLUE}设置 conda 环境: $ip${NC}"
    
    $(build_ssh_cmd $ip) "cd $PROJECT_DIR && conda env create -f environment.yml || conda env update -f environment.yml"
    
    echo -e "${GREEN}✓ Conda 环境设置完成: $ip${NC}"
}

# 主部署流程
main() {
    echo -e "${YELLOW}开始部署...${NC}"
    
    # 检查所有节点连接
    echo -e "\n${BLUE}== 步骤 1: 检查网络连接 ==${NC}"
    
    if ! check_connection $MASTER_IP; then
        echo -e "${RED}Master 节点连接失败，退出部署${NC}"
        exit 1
    fi
    
    failed_workers=()
    for worker_ip in "${WORKER_IPS[@]}"; do
        if ! check_connection $worker_ip; then
            failed_workers+=($worker_ip)
        fi
    done
    
    if [[ ${#failed_workers[@]} -gt 0 ]]; then
        echo -e "${YELLOW}警告: 以下 Worker 节点连接失败: ${failed_workers[*]}${NC}"
        read -p "是否继续部署？(y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # 创建 environment.yml
    echo -e "\n${BLUE}== 步骤 2: 生成 conda 环境配置 ==${NC}"
    create_environment_yml
    
    # 部署 Master
    echo -e "\n${BLUE}== 步骤 3: 部署 Master 节点 ==${NC}"
    deploy_files $MASTER_IP "master"
    setup_conda_env $MASTER_IP
    
    # 部署 Workers
    echo -e "\n${BLUE}== 步骤 4: 部署 Worker 节点 ==${NC}"
    for worker_ip in "${WORKER_IPS[@]}"; do
        if [[ ! " ${failed_workers[@]} " =~ " ${worker_ip} " ]]; then
            deploy_files $worker_ip "worker"
            setup_conda_env $worker_ip
        fi
    done
    
    # 生成启动说明
    echo -e "\n${BLUE}== 部署完成 ==${NC}"
    echo -e "${GREEN}✓ 所有节点部署完成${NC}"
    
    echo -e "\n${YELLOW}启动说明:${NC}"
    echo -e "1. 在 Master 节点 ($MASTER_IP) 上运行:"
    echo -e "   cd $PROJECT_DIR && ./start_master.sh"
    echo
    echo -e "2. 在 Worker 节点上分别运行:"
    local worker_id=0
    for worker_ip in "${WORKER_IPS[@]}"; do
        if [[ ! " ${failed_workers[@]} " =~ " ${worker_ip} " ]]; then
            echo -e "   Worker $worker_id ($worker_ip): cd $PROJECT_DIR && ./start_worker.sh --worker_id $worker_id --master_ip $MASTER_IP"
            ((worker_id++))
        fi
    done
    
    echo -e "\n${BLUE}========================================${NC}"
}

# 创建 conda 环境配置
create_environment_yml() {
    cat > environment.yml << 'EOF'
name: fed
channels:
  - conda-forge
  - nvidia
dependencies:
  - python=3.9
  - pip
  - numpy=1.21.0
  - scipy
  - pandas
  - openpyxl
  - flask
  - requests
  - pip:
    - tensorflow==2.12.0
    - tensorflow-gpu==2.12.0
EOF
    echo -e "${GREEN}✓ environment.yml 已生成${NC}"
}

# 运行主函数
main "$@"
