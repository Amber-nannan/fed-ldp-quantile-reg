#!/bin/bash

# 创建结果目录
RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"

# 设置不同的r值（差分隐私强度）
r_values=("0.25" "0.5" "0.75")

# 设置不同的Em模式
em_modes=("1" "5" "log")

# 运行实验
for r in "${r_values[@]}"; do
  for em in "${em_modes[@]}"; do
    echo "====================================="
    echo "Running experiment with r=$r, Em=$em"
    echo "====================================="
    
    # 运行实验并将输出保存到文件
    LOG_FILE="${RESULTS_DIR}/r_${r}_em_${em}.log"
    if [[ "$em" == "log" ]]; then
      flwr run . --run-config "r=$r local-updates-mode=\"log\"" > "$LOG_FILE" 2>&1
    else
      flwr run . --run-config "r=$r local-updates-mode=$em" > "$LOG_FILE" 2>&1
    fi
    
    # 提取最后一轮的MSE结果
    FINAL_MSE=$(grep "整体 MSE:" "$LOG_FILE" | tail -1)
    echo "r=$r, em=$em, $FINAL_MSE" | tee -a "${RESULTS_DIR}/summary.txt"
    echo "Results saved to $LOG_FILE"
    echo ""
  done
done

echo "All experiments completed. Summary available in ${RESULTS_DIR}/summary.txt"