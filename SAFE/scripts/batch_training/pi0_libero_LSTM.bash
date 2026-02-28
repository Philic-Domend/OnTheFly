GROUP_NAME=pi0diff_libero_v1
SAFE_OPENPI_ROLLOUT_ROOT=/data/huangdi/huangruixiang/SAFE/openpi/rollouts/

for REG in 1e-3; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0 \
        dataset.diff_idx_rel=1.0 \
        model=lstm \
        model.lr=1e-3 \
        model.lambda_reg=${REG} \
        train.seed=0 \
        train.exp_suffix=lstm
done