# === CINN / 动转静总开关（应用级）===
# PHYE2E_USE_CINN 控制 trainer 是否对 encoder/decoder 做 paddle.jit.to_static。
# 事实：Paddle 3.3 下 to_static 一旦包裹即强制编译 CINN，FLAGS_use_cinn 无法
# 独立关闭，故「动转静」与「CINN」合一，由本开关统一控制；不设则纯动态图。
export PHYE2E_USE_CINN=true

# 框架级 CINN / 组合算子 flag（与总开关配套，语义双保险）
export FLAGS_prim_enable_dynamic=true && export FLAGS_prim_all=true
export FLAGS_use_cinn=true

# 是否打印 Program IR 信息 (用于调试)
export FLAGS_print_ir=false

python ./train.py \
        --max_epoch 1 \
        --dump_path ./ \
        --exp_name test \
        --exp_id CINN+nvidia \
        --n_steps_per_epoch 500 \
        --print_freq 50 \
        --optimizer adam_inverse_sqrt,warmup_updates=100 \
        --collate_queue_size 500 \
        --batch_size 256 \
        --save_periodic -1 \
        --save_periodic_from 40\
        --eval_size 32 \
        --batch_size_eval 32 \
        --num_workers 0 \
        --max_len 200 \
        --max_number_bags -1 \
        --max_input_points 200 \
        --tokens_per_batch 5000 \
        --add_consts 1 \
        --device "cuda:0" \
        --use_exprs 100000 \
        --use_dimension_mask 0 \
        --expr_train_data_path "./data/exprs_train.json" \
        --expr_valid_data_path "./data/exprs_valid.json" \
        --sub_expr_train_path "./data/exprs_seperated_train.json"\
        --sub_expr_valid_path "./data/exprs_seperated_valid.json"\
        --decode_physical_units "single-seq" \
        --use_hints "units,complexity,unarys,consts" \
        --random_variables_sequence 0 \
        --max_trials 10\
        --generate_datapoints_distribution "positive,multi"\
        --rescale 0\
        --reload_model /home/lkyu/baidu/PhyE2E/models/model.pdparams
