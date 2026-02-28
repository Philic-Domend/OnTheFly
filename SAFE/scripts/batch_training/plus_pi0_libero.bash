GROUP_NAME=pi0diff_libero_v1
SAFE_OPENPI_ROLLOUT_ROOT=/data/huangdi/huangruixiang/SAFE/openpi/rollouts/

for REG in 1e-3; do
    python -m failure_prob.train_plus \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0 \
        dataset.diff_idx_rel=1.0 \
        model=winTransformer \
        model.lr=3e-5 \
        model.lambda_reg=${REG} \
        train.seed=0 \
        model.n_epochs=1000 \
        model.batch_size=1000 \
        model.if_use_vl=False
done