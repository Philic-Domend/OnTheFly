#!/usr/bin/env bash
set -e

# =========================
# 可配置参数
# =========================
POLICY=pi0_libero
SUITE_NAME=libero_90

START_N=1
END_N=5

# =========================
# 主循环
# =========================
for (( n=${START_N}; n<=${END_N}; n++ )); do
    echo "======================================"
    echo " Running iteration n=${n}"
    echo "======================================"

    # 1️⃣ 启动 policy server（后台）
    uv run scripts/serve_policy.py \
        --env LIBERO \
        --record \
        --record_suffix ${n} \
        --save_name ${POLICY}-${SUITE_NAME} \
        policy:checkpoint \
        --policy.config pi0_libero \
        --policy.dir s3://openpi-assets/checkpoints/pi0_libero \
        > serve_policy_${n}.log 2>&1 &

    SERVE_PID=$!
    echo "[INFO] serve_policy started (PID=${SERVE_PID})"

    # 等 server 启动
    sleep 10

    # 2️⃣ 运行 libero 任务（前台）
    xvfb-run -a -s "-screen 0 1024x768x24" \
        python examples/libero/main.py \
        --args.task_suite_name ${SUITE_NAME} \
        --args.save_name ${POLICY}-${SUITE_NAME} \
        --args.start-task-id ${n} \
        --args.end-task-id ${n}

    echo "[INFO] libero task ${n} finished"

    # 3️⃣ 关闭 server
    kill ${SERVE_PID}
    wait ${SERVE_PID} 2>/dev/null || true

    echo "[INFO] serve_policy stopped for n=${n}"
done

echo "======================================"
echo " All runs finished"
echo "======================================"
