一句话结论

默认脚本下，Paddle 侧主干训练/评估的神经网络实际按 fp32 跑；只有显式打开 --amp/--fp16 才进入混合精度分支，而评估与高层推理的最终结果通常又会落到 BFGS +
sympypaddle 的 float64 精炼；仓库里没有 bf16、量化、静态导出 precision mode 的真实执行路径。

入口地图

- 训练入口是 PhysicsRegressionPaddle/train.py:26，外层启动脚本主要是 PhysicsRegressionPaddle/bash/train.sh:1 和 PhysicsRegressionPaddle/bash/
  train_small_cinn.sh:1。
- 评估入口是 PhysicsRegressionPaddle/bash/eval_bash.py:22，外层脚本是 PhysicsRegressionPaddle/bash/eval_synthetic.sh:8 和 eval_feynman.sh。
- 高层推理入口是 PhysicsRegressionPaddle/PhysicsRegression.py:26。
- 导出入口分两类：模型 bundle 导出是 PhysicsRegressionPaddle/PhysicsRegression.py:74；train.py 里的 --export_data 实际走 PhysicsRegressionPaddle/
  symbolicregression/trainer.py:620，导出的是文本数据，不是推理模型。
- 配置体系不是 YAML。实际是 argparse 的 PhysicsRegressionPaddle/parsers.py:8 + env.register_args 的 PhysicsRegressionPaddle/symbolicregression/envs/
  environment.py:756 + shell 脚本传参；仓库内未发现 .yml/.yaml。

训练/评估/导出/推理精度总结表

| 环节            | 真实入口                                                    | 默认实际精度                                                                | 控制方式                                                                | 结论                                                                      |
| --------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 训练            | train.py -> Trainer.enc_dec_step                            | 神经网络主干默认 fp32；输入到网络前是 int64 token                           | --amp、--fp16、--nvidia_apex、--device、--cpu                           | 默认脚本没传 AMP 参数，所以不进 autocast；只有手动开 --amp 才有混合精度   |
| 评估            | eval_bash.py -> Evaluator.evaluate_e2e                      | 候选生成默认 fp32；最终 BFGS 精炼和符号求值走 float64                       | CLI 参数控制神经网络部分；BFGS 路径由代码默认策略决定                   | 默认评估不是纯 fp32 端到端，最终结果混入 float64 精炼                     |
| 导出            | PhyReg.save / Trainer.save_checkpoint / Trainer.export_data | 不做显式 dtype 转换；export_data 只控制文本小数位                           | float_precision 只影响文本格式；无独立 precision mode                   | 没有静态图导出或 inference precision mode                                 |
| 高层推理        | PhysicsRegression.py:PhyReg                                 | 候选生成无全局 autocast，通常按模块默认 dtype；最终常取 BFGS 结果为 float64 | PhyReg 继承模型 bundle 里的 params； refinement_strategy 间接影响后处理 | 和 eval_bash.py 不同，PhyReg 的精度状态会受保存到模型文件里的 params 影响 |
| Oracle 辅助路径 | Oracle/oracle.py                                            | 显式 float32                                                                | 无 AMP/量化开关                                                         | Oracle 是独立的纯 float32 小网络路径                                      |

所有精度控制参数/环境项总表

| 控制项                                                                                          | 声明位置                                                                        | 状态                          | 实际作用                                                                                                                |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| --fp16                                                                                          | PhysicsRegressionPaddle/parsers.py:34                                           | 真实生效                      | 影响 TransformerModel.self.dtype，并被 Trainer.init_amp() 约束；单独开无效，代码要求 fp16 => amp>=1                     |
| --amp                                                                                           | PhysicsRegressionPaddle/parsers.py:37                                           | 真实生效                      | -1 关闭 AMP；>=0 进入 init_amp 和 autocast/scaler 分支；但在 Paddle 原生路径里没有把 0/1/2/3 映射到 autocast(level=...) |
| O1/O2/O3                                                                                        | PhysicsRegressionPaddle/symbolicregression/trainer.py:236                       | 条件生效                      | 只在 --nvidia_apex=true 时通过 opt_level="O%i" 传给 Apex；原生 Paddle AMP 不 消费这个级别                               |
| --nvidia_apex                                                                                   | PhysicsRegressionPaddle/parsers.py:372                                          | 条件生效                      | 走 Apex amp.initialize/scale_loss/master_params 分支；否则走 paddle.amp.GradScaler                                      |
| --cpu / --device                                                                                | PhysicsRegressionPaddle/parsers.py:355 / PhysicsRegressionPaddle/parsers.py:550 | 真实生效                      | 控制设备，不直接指定 dtype，但会决定 AMP 路径是 否可能生效                                                              |
| --float_precision                                                                               | PhysicsRegressionPaddle/symbolicregression/envs/environment.py:756              | 真实生效，但不是 tensor dtype | 控制数字 token 化和 export_data 文本输 出的小数位                                                                       |
| --mantissa_len / --max_exponent                                                                 | PhysicsRegressionPaddle/symbolicregression/envs/environment.py:762              | 真实生效，但不是 tensor dtype | 控制数值编码粒度，不控制 Paddle 张量精度                                                                                |
| --rescale                                                                                       | PhysicsRegressionPaddle/parsers.py:43                                           | 真实生效，但不是 dtype        | 只控制 StandardScaler 预缩放，PhysicsRegressionPaddle/symbolicregression/model/ sklearn_wrapper.py:84                   |
| PADDLE_XCCL_BACKEND                                                                             | PhysicsRegressionPaddle/train.py:22                                             | 真实生效，但不是 dtype        | 指定后端为 iluvatar_gpu                                                                                                 |
| FLAGS*prim*\* / FLAGS_use_cinn / FLAGS_print_ir                                                 | PhysicsRegressionPaddle/bash/train_small_cinn.sh:2                              | 真实生效，但不是 dtype        | 控制编译器/IR，不是精度模式                                                                                             |
| bf16 / bfloat16 / pure_fp16 / pure_bf16 / multi_precision / master_weight / quant / int8 / int4 | .py 代码检索                                                                    | 未实现                        | 可执行 Python 路径里未找到真实调用                                                                                      |

关键证据

- 训练 AMP 开关的真实落点在 PhysicsRegressionPaddle/symbolicregression/trainer.py:102 和 PhysicsRegressionPaddle/symbolicregression/trainer.py:229。代码明确有
  assert params.amp >= 1 or not params.fp16，并在原生路径创建 paddle.amp.GradScaler(...)。
- 前向 autocast 的真实执行点在 PhysicsRegressionPaddle/symbolicregression/trainer.py:725。只有 else 分支用了 with paddle.cuda.amp.autocast():；默认脚本不传
  --amp，所以不会走这里。
- loss/backward/scaler 的真实路径在 PhysicsRegressionPaddle/symbolicregression/trainer.py:257。普通路径是 loss.backward(); optimizer.step()；原生 AMP 路径是
  self.scaler.scale(loss).backward()、unscale\_、step、update；Apex 路径才出现 apex.amp.master_params(self.optimizer)。
- 输入到主网络前不是浮点张量，而是 token。原始采样数据来自 PhysicsRegressionPaddle/symbolicregression/envs/generators.py:1443 的 NumPy 浮点数组；进入网络前在
  PhysicsRegressionPaddle/symbolicregression/model/embedders.py:145 和 PhysicsRegressionPaddle/symbolicregression/envs/environment.py:142 被编码成 int64。
- 模型参数没有看到显式 float16/bfloat16 构建。层定义是普通 Embedding/Linear/LayerNorm，例如 PhysicsRegressionPaddle/symbolicregression/model/transformer.py:67；
  模块加载后只在 PhysicsRegressionPaddle/symbolicregression/model/**init**.py:64 做 .to(device)，没有 .to(float16)。
- fp16 对模型内部的直接影响主要是 PhysicsRegressionPaddle/symbolicregression/model/transformer.py:185 的 self.dtype = paddle.float16 if params.fp16 else
  paddle.float32，以及生成阶段局部 .to(self.dtype)，见 PhysicsRegressionPaddle/symbolicregression/model/transformer.py:542 和 PhysicsRegressionPaddle/
  symbolicregression/model/transformer.py:611。
- 即便前面有混合精度，decoder loss 也会强制回到 float32。证据在 PhysicsRegressionPaddle/symbolicregression/model/transformer.py:475：
  cross_entropy(input=scores.float(), ...)；units loss 同样 scores_units.float()。
- 推理侧没有全局 autocast。高层候选生成在 PhysicsRegressionPaddle/symbolicregression/model/model_wrapper.py:43 只有 @paddle.no_grad()，没有 autocast() 包裹。
- 评估和高层推理最终会落到 BFGS。评估在 PhysicsRegressionPaddle/evaluate.py:185 取 refinement_type="BFGS"，并在 PhysicsRegressionPaddle/evaluate.py:194 /
  PhysicsRegressionPaddle/evaluate.py:217 用 BFGS 结果做预测；高层 PhyReg.fit() 也在 PhysicsRegressionPaddle/PhysicsRegression.py:403 到 [408] 取 BFGS 结果。
- BFGS 路径明确是 float64。见 PhysicsRegressionPaddle/symbolicregression/model/utils_wrapper.py:168 的 dtype=paddle.float64、PhysicsRegressionPaddle/
  symbolicregression/model/utils_wrapper.py:177 / [178] 的 X/y -> paddle.float64、[188] 和 [206] 的 coeffs -> paddle.float64。
- 最终符号求值不是神经网络 dtype，而是树执行 dtype。见 PhysicsRegressionPaddle/symbolicregression/model/sklearn_wrapper.py:433 和 [465]：y = node.val(X_idx)，这
  里取决于 X_idx 的 NumPy dtype。
- checkpoint / model bundle 保存加载没有仓库级显式 dtype 转换。保存见 PhysicsRegressionPaddle/symbolicregression/trainer.py:405 和 PhysicsRegressionPaddle/
  PhysicsRegression.py:74；加载见 PhysicsRegressionPaddle/symbolicregression/checkpoint_io.py:17 和 PhysicsRegressionPaddle/symbolicregression/trainer.py:457，都
  只是 paddle.load + set_state_dict。
- optimizer 没有 multi_precision/master_weight 参数路径。证据是 PhysicsRegressionPaddle/symbolicregression/optim.py:305 到 [384] 的 get_optimizer(...) 只转发常规
  optimizer 参数，代码检索也没有这些关键词命中。
- 导出没有独立 precision mode。仓库内可执行 Python 没有 paddle.jit.save、jit.to_static、save_inference_model 路径；train.py 的导出分支只是
  PhysicsRegressionPaddle/symbolicregression/trainer.py:631 里按 float_precision 格式化文本。
- 量化路径不存在。对 quant/int8/int4 的 .py 检索无命中；bf16/bfloat16/pure_fp16/pure_bf16/multi_precision/master_weight 也无 runnable Python 命中。
- 文档层确实有相关表述，但不等于运行时支持。相关命中主要在 PhysicsRegressionPaddle/PADDLE_MIGRATION.md:1211 和 PhysicsRegressionPaddle/unitTest/
  FIX_REPORT.md:104，它们是迁移/测试说明，不是入口代码。

未确认项

- 原生 paddle.cuda.amp.autocast() 没有传 dtype 或 level，所以精确的 cast policy 取决于 Paddle 运行时和后端，不是仓库代码固定死的。
- train.py 强制设了 iluvatar_gpu 后端，但训练代码调用的是 paddle.cuda.amp.autocast()；在该自定义设备上的实际 AMP 行为，需要运行时确认。
- reload_checkpoint / set_state_dict 代码层没有显式 dtype 转换，但 Paddle 框架内部在参数 dtype 不一致时是否做隐式转换，单靠静态阅读不能完全确认。
- PhyReg 与 eval_bash.py 的精度来源不同：前者继承模型文件里保存的 params，后者继承 CLI 参数；这次没有读取实际 models/model.pdparams 内容，所以没有对该文件里的
  fp16 状态做实证。

最小验证方案

- cd "/home/lkyu/baidu/PhyE2E/PhysicsRegressionPaddle" && python "./train.py" --max_epoch 1 --n_steps_per_epoch 1 --batch_size 2 --num_workers 0 --device
  "iluvatar_gpu:0" --eval_in_domain false
  思路：在 Trainer.enc_dec_step 和 optimize 断点/打印 params.amp、self.scaler is not None、loss.dtype、首个参数 dtype，确认默认训练是否为非 AMP、主干是否为
  fp32。
- cd "/home/lkyu/baidu/PhyE2E/PhysicsRegressionPaddle" && python "./train.py" --max_epoch 1 --n_steps_per_epoch 1 --batch_size 2 --num_workers 0 --device
  "iluvatar_gpu:0" --eval_in_domain false --fp16 true --amp 1
  思路：确认进入 autocast + GradScaler 分支，同时检查 scores.float() 后 loss 是否仍回到 float32。
- cd "/home/lkyu/baidu/PhyE2E/PhysicsRegressionPaddle" && python "./bash/eval_bash.py" --reload_model "../models/model.pdparams" --eval_size 2 --batch_size_eval
  后半段 float64 精炼”的分层精度。
- cd "/home/lkyu/baidu/PhyE2E/PhysicsRegressionPaddle" && python - <<'PY'\nfrom PhysicsRegression import PhyReg\nm = PhyReg(\"../models/model.pdparams\",
  device=\"iluvatar_gpu:0\")\nprint(m.params.fp16)\nm.save(\"../models/model.inspect.pdparams\")\nPY
  思路：确认 PhyReg 是从模型 bundle 继承 params，并检查 save/load 前后 state_dict dtype 是否保持一致。
- cd "/home/lkyu/baidu/PhyE2E" && rg -n -g '\*.py' \"\\b(bf16|bfloat16|pure_fp16|pure_bf16|multi_precision|master_weight|quant|int8|int4)\\b\" \"./
  PhysicsRegressionPaddle\"
  思路：做一次回归检索，确认没有遗漏的低精度/量化路径。
