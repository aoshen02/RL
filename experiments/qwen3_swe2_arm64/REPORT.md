# 单节点 GB200(ARM64)跑通 NeMo-RL 旗舰 SWE2 recipe

一个 GB200 节点(4 GPU,aarch64)端到端跑通:
`NeMo-RL(Megatron)→ NeMo-Gym → vLLM → OpenHands → ARM64 SWE 沙盒 → SWE-bench 评测 → reward → replay buffer → 一个训练步`。数字均可追溯到 `agent_run/results/` 下的运行目录。

## 改动(核心修复在 Gym,不在 NeMo-RL)

**阻断性 bug**:`openhands.sh` 硬编码下载 `jq-linux-amd64`。aarch64 上下载成功、setup 全绿,但该二进制拷进 SWE 容器后 `Exec format error`;启动脚本查询返回空并 `exit 1`,而它是被 **source** 的,于是杀死调用者 shell。OpenHands 只看到 `command timed out after 600.0 seconds`,重试 3 次 = 每条 rollout 白等 1800s。**表面是超时,真实故障在第 0 秒。** 改法:按 `uname -m` 分发;二进制跑不起来就重下;去掉 `jq --version || true` —— 正是它把错误吞掉。

**优化**:`NEMO_GYM_SWE_NO_COPY=1` 把 `cp -al /testbed` 换成 `ln -s`。容器可写层是 RAM-backed,递归复制费时费内存。安全性来自 `--writable-tmpfs`(SIF 只读共享,每容器一份内存 overlay)—— 实测同题两条并发 rollout 产出了不同 patch。 分支:`aoshen02/Gym` @ `fix/swe-arm64-jq`(`openhands.sh` +17 −4,`swe_agents/app.py` +62)· `aoshen02/RL` @ `review/qwen3-swe2-singlenode`(脚手架/文档/证据)。

## 配置与数据

`Qwen3-30B-A3B-Thinking-2507`(MoE)· recipe `grpo_qwen3_30ba3b_thinking_swe2.yaml` · 训练 Megatron TP2/PP1/CP1/EP1(optimizer 全量 CPU offload)· 生成 vLLM TP2 non-colocated async,`max_model_len=32768` · Agent 是 **OpenHands**(Gym 默认,`CodeActAgent`,200 轮上限),评测是官方 `swebench.harness.run_local_evaluation`,两者完全分离。数据:`SWE-bench_Verified` 的 **10 个 django 题**,按金 patch 规模分层(1~29 行),不是全挑简单题;每题一个 ARM64 SIF,由 `docker://swebench/sweb.eval.arm64.<id>` 零定制转换。GRPO:10 prompt × 2 采样 = 20 条 rollout,`max_num_steps=1`,沙盒并发 10。

## 结果

| Job | 代码 | reward mean | resolved |
|---|---|---|---|
| 12946 | 精简前 | 0.4500 | 9/20 |
| 13203 | 精简后 | 0.3000 | 6/20 |
| 13206 | 精简后 | **0.5000** | 10/20 |

三次均 `exit_code=0`。精简前的 0.45 落在两次精简后**中间**,故无证据表明精简改变行为;只看 13203 会误判为退化(n=20,z≈1.0,采样波动)。通过率随金 patch 规模**单调下降**:1–2 行 6/8,5 行 2/4,11–29 行 2/8。**非零 advantage** `±0.7071, std=0.3162` —— 组内有对有错才有梯度;早期两跑分别是单样本(`std=nan`)与 8/8 全对(`std=0`),都是零更新。冷启动全程 14m53s。

**reward 不是 hack**:① agent 与评测容器挂的 `/root/dataset/data.jsonl` 是两个不同文件,agent 那份 925B、`patch`/`test_patch`/`FAIL_TO_PASS`/`PASS_TO_PASS` 全 ABSENT;② 镜像里没有目标测试,20 条 rollout `touched_tests=0`;③ `eval.sh` 先 `git checkout <base_commit> tests/...` 再打 `test_patch`,源码侧只应用**模型的** patch;④ 一条缩进损坏的 patch 被正确判 `resolved=False`。难度单调下降本身也是泄漏做不到的。

## 复现

```bash
git clone https://github.com/aoshen02/RL  -b review/qwen3-swe2-singlenode
git clone https://github.com/aoshen02/Gym -b fix/swe-arm64-jq <RL>/3rdparty/Gym-workspace/Gym
cd <RL>/experiments/qwen3_swe2_arm64
bash   scripts/build_swe_sif_arm64.sh   django__django-17029          # 造 SIF,~2 分钟
python scripts/make_swe2_dataset_row.py django__django-17029 data/django17029_swe2.jsonl
bash   scripts/prewarm_nemo_gym_qwen3_swe2.sh                          # 预热 venv,须串行

SWE_DATASET=/workspace/agent_run/data/qwen3_swe2_smoke/django_mix10_swe2.jsonl \
NUM_PROMPTS=10 NUM_GENERATIONS=2 SWE_CONCURRENCY=10 \
  bash scripts/launch_qwen3_swe2_tp2_e2e.sh
bash handoff_qwen3_swe2_cold_start.sh --verify [run_dir]   # 0=全过 1=有必需项未过 2=用法错误
```

前置:Slurm + pyxis/enroot,1 节点 4 GPU(aarch64),镜像 `nemo-rl-vllm-latest.sqsh`。dockerhub 上 arm64 只覆盖**部分** instance(django 约 141 个),不存在的返回 401。可选:`SWE_INSTANCE=` 单题 · `GYM_DIR=` 换代码源而不动主工作区 · `SWE_VERIFY_GOLDEN=1` 用金 patch 校准评测链路(**校准非交付**)。并发取 10 而非 20:沙盒实测 1.1–1.6 GB/个,但 `apptainer_memory_limit_mb=32768` 在 enroot 里**无 cgroup 硬限**、仅 watchdog 软执行,最坏是 `并发 × 32 GB`;内存大头在训练侧,节点 893 GB 驻留后约剩 185 GB。

## 边界

**支撑的结论只有一句:上述链路完整跑通且 `exit_code=0`。** 它不证明:不是模型能力评估(10 题 × 2 采样);不是训练有效性验证(`max_num_steps=1`,无收敛、无多步);证据链 item 3/4/5/8 **只覆盖 20 条 rollout 中的 1 条**(收集器取 `head -1`),只有聚合项覆盖全部;item 6 恒为 NO 且被豁免(OpenHands 自身 `output.jsonl` 恒 0 字节,轨迹改由 item 7 从 `train_data_step*.jsonl` 解码);**只在 aarch64 验证过,合上游前需 x86 回归**。 已知偶发:job 13204 两个 vLLM TP worker 被系统杀掉(`SYSTEM_ERROR` → `Executor failed.`),同节点同代码重跑通过,判为基础设施偶发。Slurm 的 `batch` 分区是 `OverSubscribe=EXCLUSIVE`,每作业独占整节点 —— N 个小作业 = N 倍墙钟。详见 `ROOT_CAUSE_swe_entry_600s_timeout.md`(jq 推理链,含两次自我更正)· `ROOT_CAUSE_vllm_enginedead_keyerror.md`(`mamba_cache_mode` 继承链陷阱,含一处撤回的错误归因)· `REWARD1_EVIDENCE.md`(含「被证伪的假设」9 条)· `AUDIT.md`(独立审计,含校验器自身的假阳性与假阴性)。
