# 在单节点 GB200 上跑通 NeMo-RL 旗舰 SWE2 recipe(ARM64)

本报告记录:做了什么、用什么模型和数据、在 NeMo-RL 中拿到什么结果、以及别人如何复现。
所有数字都可追溯到 `agent_run/results/` 下的运行目录,报告不含未经日志核对的结论。

---

## 1. 做了什么

在**一个 GB200 节点(4×GPU,aarch64)**上,把 NeMo-RL 的旗舰 NeMo-Gym SWE2 recipe
端到端跑通:

```
NeMo-RL(Megatron)→ NeMo-Gym → vLLM → OpenHands → ARM64 SWE 沙盒
  → SWE-bench 评测 → reward → replay buffer → 一个训练步
```

过程中修复了一个**上游在 ARM64 上的阻断性缺陷**,并新增了一个冷启动优化。

### 1.1 修复:ARM64 上 jq 架构不符(阻断性)

`setup_scripts/openhands.sh` 硬编码下载 `jq-linux-amd64`。在 aarch64 上**下载会成功**,
所以 setup 全绿;但该二进制会被拷进 SWE 容器,在那里 `Exec format error`。
`instance_swe_entry.sh` 的 instance 查询因此返回空并命中 `exit 1`,而该脚本是被
**source** 的,于是连带杀死调用者的 shell。OpenHands 只能观察到:

```
Failed to source /swe_util/instance_swe_entry.sh: command timed out after 600.0 seconds
```

重试 3 次 = 每条 rollout 白等 1800 秒。**表面是超时,真实故障在第 0 秒。**

修法三处:按 `uname -m` 分发;现有二进制无法执行时重新下载(让已污染的树自愈);
去掉 `jq --version || true` —— 正是这个 `|| true` 把错误吞掉,使问题拖到沙盒内才爆发。

### 1.2 新增:`NEMO_GYM_SWE_NO_COPY=1`(冷启动优化,非缺陷修复)

上游默认 `cp -al /testbed`(硬链接失败则 `cp -r`)。容器可写层是 **RAM-backed** 的,
所以递归复制既是时间成本也是内存成本,并发沙盒越多越明显。该开关改为 `ln -s /testbed`。

两处**刻意不动**并写进注释:`conda activate testbed`(它负责运行 `activate.d/*.sh`)
和 `run_infer.py` 的 `source ~/.bashrc`(它把 conda 放进 PATH)。早期版本跳过它们,
导致 jq 和 sleep 一起从 PATH 消失,是同一个 600s 超时的另一个来源。

### 1.3 代码位置

| 仓库 | 分支 | 改动 |
|---|---|---|
| `aoshen02/Gym` | `fix/swe-arm64-jq` (`77cdaa31`) | `swe_agents/setup_scripts/openhands.sh` +17 −4;`swe_agents/app.py` +62 |
| `aoshen02/RL` | `review/qwen3-swe2-singlenode` (`dbfbdd78d`) | 实验脚手架、文档、证据 |

**核心修复在 Gym,不在 NeMo-RL。** NeMo-RL 侧只有一个单节点 recipe yaml 的形状适配,
而且它的 5 个键全部被 run 脚本的 CLI override 覆盖,属于冗余,可删。

---

## 2. 模型、数据、配置

### 2.1 模型与并行

| | |
|---|---|
| 模型 | `Qwen3-30B-A3B-Thinking-2507`(MoE) |
| Recipe | `examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe2.yaml` |
| 训练 | Megatron TP2 / PP1 / CP1 / EP1,optimizer 全量 CPU offload |
| 生成 | vLLM TP2,**non-colocated**,async engine |
| 序列长度 | `max_total_sequence_length=32768`,`max_model_len=32768` |
| Agent | OpenHands(NV fork `sdevare-nv/nv-OpenHands` @ `5f01800`),`CodeActAgent`,`agent_max_turns=200` |
| 集群 | 1 节点 × 4 GPU(GB200,aarch64) |
| 镜像 | `nemo-rl-vllm-latest.sqsh`(enroot),容器内 apptainer 1.4.5 |

Agent harness 是 **OpenHands**,这也是 Gym 的默认值
(`agent_framework: Literal["openhands","opencode"] = "openhands"`),不是
mini-swe-agent / Claude Code / Codex。评测侧是官方 `swebench.harness.run_local_evaluation`,
与 agent 侧完全分离。

### 2.2 数据集

`princeton-nlp/SWE-bench_Verified` 的 **10 个 django instance**,按金 patch 规模**分层挑选**,
不是全挑简单题:

| instance | django | gold 行 | F2P | P2P |
|---|---|---|---|---|
| `django__django-16429` | 4.2 | 1 | 4 | 21 |
| `django__django-17029` | 5.0 | 1 | 1 | 43 |
| `django__django-16485` | 5.0 | 2 | 1 | 9 |
| `django__django-16333` | 4.2 | 2 | 1 | 87 |
| `django__django-14539` | 4.0 | 5 | 2 | 14 |
| `django__django-16662` | 5.0 | 5 | 1 | 51 |
| `django__django-14999` | 4.1 | 11 | 1 | 113 |
| `django__django-15916` | 4.2 | 11 | 2 | 149 |
| `django__django-15128` | 4.1 | 27 | 1 | 282 |
| `django__django-12713` | 3.1 | 29 | 1 | 58 |

每题一个 ARM64 SIF,由 `docker://swebench/sweb.eval.arm64.<id>` 直接转换,**零定制**
(已用 `unsquashfs` 验证:镜像内只有 `/opt/miniconda3`、`/testbed`、`/root/setup_*.sh`;
`/swe_util` 和 `/root/dataset` 都是运行时 bind 进去的)。

GRPO:`num_prompts_per_step=10`,`num_generations_per_prompt=2`,
`train_global_batch_size=20`,`max_num_steps=1`,沙盒并发 10。

---

## 3. 结果

### 3.1 三次同配置运行

| Job | 代码 | reward mean | resolved | exit |
|---|---|---|---|---|
| 12946 | 精简前(主工作区) | 0.4500 | 9/20 | 0 |
| 13203 | 精简后(独立 worktree) | 0.3000 | 6/20 | 0 |
| 13206 | 精简后(独立 worktree) | **0.5000** | 10/20 | 0 |

精简前的 0.45 落在两次精简后运行的**中间**,因此没有证据表明代码精简改变了行为。
只看 13203 的 0.30 会误判为退化 —— n=20 下这是采样波动(z≈1.0)。

### 3.2 13206 逐题结果

```
gold 1–2 行     6/8
gold 5 行       2/4
gold 11–29 行   2/8      ← 通过率随金 patch 规模单调下降
```

| instance | gold 行 | rollout |
|---|---|---|
| `django__django-16429` | 1 | ✅ ✅ |
| `django__django-17029` | 1 | ✅ ✅ |
| `django__django-16485` | 2 | ❌ ❌ |
| `django__django-16333` | 2 | ✅ ✅ |
| `django__django-14539` | 5 | ❌ ❌ |
| `django__django-16662` | 5 | ✅ ✅ |
| `django__django-14999` | 11 | ✅ ❌ |
| `django__django-15916` | 11 | ❌ ❌ |
| `django__django-15128` | 27 | ❌ ❌ |
| `django__django-12713` | 29 | ✅ ❌ |

**非零 advantage**:`min=-0.7071, max=+0.7071, std=0.3162`。组内有对有错才产生梯度;
早期两次运行分别是单样本(`std=nan`)和 8/8 全对(`std=0`),都是零更新。

### 3.3 reward 不是 hack —— 四条独立证据

1. **金 patch 未泄漏**:agent 容器与评测容器挂载的 `/root/dataset/data.jsonl` 是**两个不同文件**。
   agent 那份 925 字节,`patch` / `test_patch` / `FAIL_TO_PASS` / `PASS_TO_PASS` **全部 ABSENT**;
   评测那份 5002 字节才完整。
2. **测试代码未泄漏**:`unsquashfs` 直读镜像,目标测试不存在;20 条 rollout 的工具调用记录中
   `touched_tests=0`。
3. **评测基于 base_commit 而非最新代码**:`eval.sh` 显式
   `git checkout <base_commit> tests/...` 后再 `git apply test_patch`。源码侧只应用**模型的** patch,
   金 patch 从未被应用。镜像 HEAD 与 base_commit 不同是因为 SWE-bench 造镜像时自建了一个
   commit(`Author: SWE-bench <setup@swebench.config>`),代码树即 base_commit 状态。
4. **评测会判错**:同一 instance、同一评测路径,一条缩进损坏的 patch 被判
   `resolved=False`、`FAIL_TO_PASS failure=[...]`。

通过率随难度单调下降,是泄漏奖励**无法**产生的模式 —— 泄漏不认得金 patch 有多少行。

### 3.4 冷启动分段耗时(13206)

```
container_ready       start
gym_venv_verified     +7.4s      ← venv 预热复用
nemo_rl_launch        +0.0s
nemo_rl_exit          +823.7s    ← 全程 14m53s
```

Gym 5 个 server 约 3 分钟就绪;模型加载 28.51 GiB;单个沙盒
`initialize_runtime` 41.55s、agent 峰值 RSS 1.1–1.6 GB、评测峰值 126–234 MB。

---

## 4. 复现

### 4.1 前置

```
Slurm + pyxis/enroot;一个 GB200 节点(4 GPU,aarch64)
镜像  /mnt/lustre01/users/inf-aoshen/enroot/containers/nemo-rl-vllm-latest.sqsh
模型  /mnt/lustre01/users/inf-aoshen/models/Qwen3-30B-A3B-Thinking-2507
```

代码:
```bash
git clone https://github.com/aoshen02/RL  -b review/qwen3-swe2-singlenode
git clone https://github.com/aoshen02/Gym -b fix/swe-arm64-jq \
    <RL>/3rdparty/Gym-workspace/Gym
```
实验脚手架在 `experiments/qwen3_swe2_arm64/`。

### 4.2 造 ARM64 SIF 与数据行(每题一次)

```bash
bash scripts/build_swe_sif_arm64.sh django__django-17029        # ~2 分钟
python scripts/make_swe2_dataset_row.py django__django-17029 \
       data/django17029_swe2.jsonl
```

镜像来自 `docker://swebench/sweb.eval.arm64.<repo>_1776_<name>`。
**注意 dockerhub 上 arm64 只覆盖一部分 instance**(django 约 141 个),不存在的会返回 401。
多题合并成一行一条的 jsonl 即可。

### 4.3 预热 Gym venv(一次)

```bash
bash scripts/prewarm_nemo_gym_qwen3_swe2.sh
```
必须串行。多个 Gym server 并发跑 `uv pip install` 会抢同一把 uv 锁并超时。

### 4.4 运行

```bash
# 单题
SWE_INSTANCE=django__django-17029 NUM_GENERATIONS=8 SWE_CONCURRENCY=4 \
  bash scripts/launch_qwen3_swe2_tp2_e2e.sh

# 10 题 × group 2(本报告的配置)
SWE_DATASET=/workspace/agent_run/data/qwen3_swe2_smoke/django_mix10_swe2.jsonl \
NUM_PROMPTS=10 NUM_GENERATIONS=2 SWE_CONCURRENCY=10 \
  bash scripts/launch_qwen3_swe2_tp2_e2e.sh
```

可选:`GYM_DIR=<另一份 Gym 树>` 把代码源指向别处而不动主工作区(会自动 bind
所有 `swe_*_setup` 构建产物复用);`SWE_VERIFY_GOLDEN=1` 跳过 agent、直接用金 patch
校准评测链路(**是校准不是交付**)。

### 4.5 验证

```bash
bash handoff_qwen3_swe2_cold_start.sh --verify [run_dir]
#   0 = 必需项全部通过   1 = 有必需项未通过   2 = 用法/环境错误
```

它会导出轨迹(用 tokenizer 解 `token_ids` + `token_loss_mask`),收集证据链并给出退出码。

### 4.6 并发与内存

沙盒实测峰值 1.1–1.6 GB,10 路约 17 GB。但 `apptainer_memory_limit_mb=32768`
在 enroot 里**没有 cgroup 硬限**、只靠 watchdog 软执行(见 `app.py` 注释
"cgroups are unavailable in the enroot sandbox"),最坏口径是 `并发 × 32 GB`。
内存大头在训练侧(optimizer 全量 CPU offload,几百 GB 级),节点 893 GB 中
训练驻留后约剩 185 GB。据此把并发定在 10 而非 20。

---

## 5. 边界与已知问题

**这次验证支撑的结论只有一句**:上述链路完整跑通且 `exit_code=0`。它**不**证明:

- **不是模型能力评估**。reward mean 0.5 只覆盖 10 个 django instance、每题 2 次采样。
- **不是训练有效性验证**。`max_num_steps=1`,没有收敛、没有多步、没有指标改善。
- **证据链中 item 3/4/5/8 只覆盖 20 条 rollout 中的 1 条**(收集器取 `head -1`)。
  聚合类项(reward / buffer / step / exit code)才覆盖全部。
- **item 6 恒为 NO 且被豁免**:OpenHands 自己的 `output.jsonl` 在这条路径上永远 0 字节,
  轨迹由 item 7 从 `train_data_step*.jsonl` 解码得到。
- **只在 aarch64 上验证过**。合入上游前需要 x86 路径回归。

**已知偶发**:job 13204 出现两个 vLLM TP worker 被系统杀掉
(`Worker exit type: SYSTEM_ERROR` → `RuntimeError: Executor failed.`),同节点同代码
重跑通过,判为基础设施偶发,非代码缺陷。

**Slurm 特性**:`batch` 分区是 `OverSubscribe=EXCLUSIVE`,每个作业独占整节点。
提交 N 个小作业 = N 倍墙钟;造多个 SIF 应塞进一个 sbatch 内并行。

---

## 6. 相关文档

| 文件 | 内容 |
|---|---|
| `ROOT_CAUSE_swe_entry_600s_timeout.md` | ARM64 jq 的完整推理链,含两次自我更正 |
| `ROOT_CAUSE_vllm_enginedead_keyerror.md` | `mamba_cache_mode` 的 recipe 继承链陷阱,以及一处被撤回的错误归因 |
| `REWARD1_EVIDENCE.md` | reward=1 的证据 + 附录「被证伪的假设」9 条 |
| `AUDIT.md` | 独立审计,含校验器自身的假阳性与假阴性 |
