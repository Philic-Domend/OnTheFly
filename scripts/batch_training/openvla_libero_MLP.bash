GROUP_NAME=openvla_libero_v2
SAFE_OPENVLA_ROLLOUT_ROOT=/data/huangdi/huangruixiang/SAFE/openvla/rollouts/

for SUITE_NAME in 10; do
for SEED in 0; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_libero_${SUITE_NAME} \
        dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
        dataset.token_idx_rel=mean \
        dataset.load_to_cuda=False \
        model=indep \
        model.batch_size=64 \
        model.lr=1e-4 \
        model.lambda_reg=1e-3 \
        train.seed=${SEED} \
        train.exp_suffix=mlp
done
done