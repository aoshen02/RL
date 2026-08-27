# 单节点 GB200(ARM64)跑通 NeMo-RL 旗舰 SWE2 recipe

数字均可追溯到 `agent_run/results/` 下的运行目录。

## 1. 做了什么

一个 GB200 节点(4 GPU,aarch64)上端到端跑通:
`NeMo-RL(Megatron)→ NeMo-Gym → vLLM → OpenHands → ARM64 SWE 沙盒 → SWE-bench 评测 → reward → replay buffer → 一个训练步`

**修复(阻断性)**:`setup_scripts/openhands.sh` 硬编码 `jq-linux-amd64`。aarch64 上下载会成功、setup 全绿,但该二进制被拷进 SWE 容器后 `Exec format error`;`instance_swe_entry.sh` 的查询返回空并 `exit 1`,而它是被 **source** 的,于是杀死调用者 shell。OpenHands 只看到 `command timed out after 600.0 seconds`,重试 3 次 = 每条 rollout 白等 1800s。**表面是超时,真实故障在第 0 秒。**
改法:按 `uname -m` 分发;现有二进制无法执行时重下(让已污染的树自愈);去掉 `jq --version || true` —— 正是它把错误吞掉,使问题拖到沙盒内才爆发。

**新增(优化,非缺陷)**:`NEMO_GYM_SWE_NO_COPY=1` 把 `cp -al /testbed` 换成 `ln -s`。容器可写层是 RAM-backed,递归复制既费时也费内存。`conda activate testbed` 与 `run_infer.py` 的 `source ~/.bashrc` **刻意保留**并写进注释 —— 早期版本跳过它们导致 jq/sleep 一起掉出 PATH,是同一个 600s 超时的另一来源。

| 仓库 | 分支 | 改动 |
|---|---|---|
| `aoshen02/Gym` | `fix/swe-arm64-jq` (`77cdaa31`) | `openhands.sh` +17 −4;`swe_agents/app.py` +62 |
| `aoshen02/RL` | `review/qwen3-swe2-singlenode` | 实验脚手架、文档、证据 |

**核心修复在 Gym,不在 NeMo-RL。** NeMo-RL 侧只有一个单节点 recipe yaml,其 5 个键全被 run 脚本的 CLI override 覆盖,属冗余可删。

## 2. 模型与数据

| | |
|---|---|
| 模型 | `Qwen3-30B-A3B-Thinking-2507`(MoE) |
| Recipe | `examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe2.yaml` |
| 训练 | Megatron TP2/PP1/CP1/EP1,optimizer 全量 CPU offload |
| 生成 | vLLM TP2,non-colocated,async engine,`max_model_len=32768` |
| Agent | OpenHands(NV fork `sdevare-nv/nv-OpenHands` @ `5f01800`),`CodeActAgent`,`agent_max_turns=200` |
| 集群 | 1 节点 × 4 GPU;镜像 `nemo-rl-vllm-latest.sqsh`,容器内 apptainer 1.4.5 |

Agent harness 是 **OpenHands**(Gym 默认值),不是 mini-swe-agent / Claude Code / Codex。评测侧是官方 `swebench.harness.run_local_evaluation`,与 agent 侧完全分离。

数据集:`princeton-nlp/SWE-bench_Verified` 的 **10 个 django instance**,按金 patch 规模**分层挑选**(1~29 行,P2P 9~282),不是全挑简单题。每题一个 ARM64 SIF,由 `docker://swebench/sweb.eval.arm64.<id>` 直接转换、**零定制**(`unsquashfs` 已验证镜像内只有 `/opt/miniconda3`、`/testbed`、`/root/setup_*.sh`;`/swe_util` 与 `/root/dataset` 是运行时 bind 的)。

GRPO:`num_prompts_per_step=10`,`num_generations_per_prompt=2`,`train_global_batch_size=20`,`max_num_steps=1`,沙盒并发 10。

## 3. 结果

三次同配置运行:

| Job | 代码 | reward mean | resolved | exit |
|---|---|---|---|---|
| 12946 | 精简前 | 0.4500 | 9/20 | 0 |
| 13203 | 精简后 | 0.3000 | 6/20 | 0 |
| 13206 | 精简后 | **0.5000** | 10/20 | 0 |

精简前的 0.45 落在两次精简后运行**中间**,故无证据表明精简改变了行为。只看 13203 的 0.30 会误判为退化 —— n=20 下这是采样波动(z≈1.0)。

13206 逐题(✅=resolved):

| instance | gold 行 | rollout | | instance | gold 行 | rollout |
|---|---|---|---|---|---|---|
| `django-16429` | 1 | ✅✅ | | `django-16662` | 5 | ✅✅ |
| `django-17029` | 1 | ✅✅ | | `django-14999` | 11 | ✅❌ |
| `django-16485` | 2 | ❌❌ | | `django-15916` | 11 | ❌❌ |
| `django-16333` | 2 | ✅✅ | | `django-15128` | 27 | ❌❌ |
| `django-14539` | 5 | ❌❌ | | `django-12713` | 29 | ✅❌ |

通过率随金 patch 规模**单调下降**:1–2 行 6/8,5 行 2/4,11–29 行 2/8。
**非零 advantage**:`min=-0.7071, max=+0.7071, std=0.3162` —— 组内有对有错才有梯度;早期两跑分别是单样本(`std=nan`)和 8/8 全对(`std=0`),都是零更新。

**reward 不是 hack,四条独立证据**:
1. agent 容器与评测容器挂载的 `/root/dataset/data.jsonl` 是**两个不同文件** —— agent 那份 925B,`patch`/`test_patch`/`FAIL_TO_PASS`/`PASS_TO_PASS` 全 ABSENT;评测那份 5002B 才完整。
2. `unsquashfs` 直读镜像,目标测试不存在;20 条 rollout 的工具调用 `touched_tests=0`。
3. `eval.sh` 显式 `git checkout <base_commit> tests/...` 后再 `git apply test_patch`;源码侧只应用**模型的** patch。镜像 HEAD 与 base_commit 不同,是因为 SWE-bench 造镜像时自建了一个 commit(`Author: SWE-bench <setup@swebench.config>`),代码树即 base_commit 状态。
4. 同一 instance、同一评测路径,一条缩进损坏的 patch 被判 `resolved=False`。

通过率随难度单调下降是泄漏奖励**无法**产生的模式 —— 泄漏不认得金 patch 有多少行。

冷启动(13206,全程 14m53s):`gym_venv_verified +7.4s`(venv 预热复用)→ Gym 5 个 server 约 3 分钟就绪 → 模型加载 28.51 GiB → 单沙盒 `initialize_runtime` 41.55s。

## 4. 复现

前置:Slurm + pyxis/enroot;1 个 GB200 节点(4 GPU,aarch64);镜像 `nemo-rl-vllm-latest.sqsh`;模型 `Qwen3-30B-A3B-Thinking-2507`。

```bash
git clone https://github.com/aoshen02/RL  -b review/qwen3-swe2-singlenode
git clone https://github.com/aoshen02/Gym -b fix/swe-arm64-jq <RL>/3rdparty/Gym-workspace/Gym
cd <RL>/experiments/qwen3_swe2_arm64

# 1) 造 SIF + 数据行(每题一次,~2 分钟)
bash   scripts/build_swe_sif_arm64.sh   django__django-17029
python scripts/make_swe2_dataset_row.py django__django-17029 data/django17029_swe2.jsonl

# 2) 预热 Gym venv(一次,必须串行:并发会抢同一把 uv 锁并超时)
bash scripts/prewarm_nemo_gym_qwen3_swe2.sh

# 3) 运行(本报告的配置)
SWE_DATASET=/workspace/agent_run/data/qwen3_swe2_smoke/django_mix10_swe2.jsonl \
NUM_PROMPTS=10 NUM_GENERATIONS=2 SWE_CONCURRENCY=10 \
  bash scripts/launch_qwen3_swe2_tp2_e2e.sh

# 4) 验证(0=全过 1=有必需项未过 2=用法/环境错误)
bash handoff_qwen3_swe2_cold_start.sh --verify [run_dir]
```

dockerhub 上 arm64 只覆盖**一部分** instance(django 约 141 个),不存在的返回 401。
可选开关:`SWE_INSTANCE=<id>` 单题;`GYM_DIR=<另一份 Gym 树>` 换代码源而不动主工作区(自动 bind 所有 `swe_*_setup` 构建产物复用);`SWE_VERIFY_GOLDEN=1` 跳过 agent 用金 patch 校准评测链路(**是校准不是交付**)。

并发与内存:沙盒实测峰值 1.1–1.6 GB,10 路约 17 GB。但 `apptainer_memory_limit_mb=32768` 在 enroot 里**无 cgroup 硬限**、仅靠 watchdog 软执行,最坏口径是 `并发 × 32 GB`。内存大头在训练侧(optimizer 全量 CPU offload),节点 893 GB 中训练驻留后约剩 185 GB —— 据此把并发定在 10 而非 20。

## 5. 边界与已知问题

**支撑的结论只有一句**:上述链路完整跑通且 `exit_code=0`。它**不**证明:

- **不是模型能力评估** —— 10 个 django instance × 每题 2 次采样。
- **不是训练有效性验证** —— `max_num_steps=1`,无收敛、无多步、无指标改善。
- **证据链 item 3/4/5/8 只覆盖 20 条 rollout 中的 1 条**(收集器取 `head -1`);只有聚合项(reward/buffer/step/exit code)覆盖全部。
- **item 6 恒为 NO 且被豁免** —— OpenHands 自身的 `output.jsonl` 在这条路径上永远 0 字节,轨迹由 item 7 从 `train_data_step*.jsonl` 解码得到。
- **只在 aarch64 验证过**,合入上游前需 x86 回归。

**已知偶发**:job 13204 出现两个 vLLM TP worker 被系统杀掉(`SYSTEM_ERROR` → `RuntimeError: Executor failed.`),同节点同代码重跑通过,判为基础设施偶发。
**Slurm 特性**:`batch` 分区 `OverSubscribe=EXCLUSIVE`,每作业独占整节点 —— 提交 N 个小作业 = N 倍墙钟,造多个 SIF 应塞进一个 sbatch 内并行。

## 6. 相关文档

`ROOT_CAUSE_swe_entry_600s_timeout.md`(ARM64 jq 完整推理链,含两次自我更正)·
`ROOT_CAUSE_vllm_enginedead_keyerror.md`(`mamba_cache_mode` recipe 继承链陷阱,含一处撤回的错误归因)·
`REWARD1_EVIDENCE.md`(reward=1 证据 + 附录「被证伪的假设」9 条)·
`AUDIT.md`(独立审计,含校验器自身的假阳性与假阴性)
