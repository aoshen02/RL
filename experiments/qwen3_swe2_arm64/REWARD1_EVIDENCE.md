# reward = 1 达成,且已验证"合理"

结论所依据的两次运行都在盘上,`exit_code=0`,`handoff --verify` 12 项全过。

## 1. reward = 1 确实拿到了

| run | job | 配置 | Rewards stats | verify |
|---|---|---|---|---|
| `qwen3_swe2_tp2_e2e_20260813T105020Z` | 12930 | 单题 17029,group 8 | `min=1.0000 max=1.0000 mean=1.0000` | PASS, exit 0 |
| `qwen3_swe2_tp2_e2e_20260813T112753Z` | 12946 | 10 题混合,group 2 | `min=0.0000 max=1.0000 mean=0.4500` | PASS, exit 0 |

12946 里 9 条 rollout 拿到 reward=1,跨 **7 个不同 instance**(17029×2、16333×2、16429、14999、16662、16485、12713)。
每一条的 `report.json` 都满足:`FAIL_TO_PASS` 无 failure、`PASS_TO_FAIL` 无 failure、
`patch_successfully_applied=true`。即真的把原本失败的测试修成通过,且没有破坏任何回归测试
(最多的一条同时保住 113 个 `PASS_TO_PASS`)。

## 2. 为什么这个 reward 是"合理"的,不是 reward hack

怀疑的起点是合理的:12930 单题 8/8 全对本身就可疑。12946 就是为此设计的对照 ——
10 题按金 patch 难度分层(1~29 行,`PASS_TO_PASS` 9~282),group size 降到 2。

### 2.1 评测有区分力(不是全 1.0)

20 条 rollout:**9 对 11 错,通过率 45%**。同一题两条采样出现一对一错的有 5 题
(`16429`/`16485`/`16662`/`14999`/`12713`),这是"奖励泄漏"不可能产生的。

### 2.2 通过率随难度单调下降

| 金 patch 代码行数 | 通过率 |
|---|---|
| 1–2 行(4 题 / 8 条) | 6/8 = **75%** |
| 5 行(2 题 / 4 条) | 1/4 = 25% |
| 11–29 行(4 题 / 8 条) | 2/8 = 25% |

泄漏的奖励不认得金 patch 有多少行。最难的 `15128`(27 行 / 282 回归测试)与
`15916`(11 行 / 149)都是 0/2。

### 2.3 三条泄漏通道逐条排除

| 通道 | 检查方式 | 结论 |
|---|---|---|
| 金 patch | 对比 agent 容器与评测容器各自挂载的 `/root/dataset/data.jsonl` | agent 那份 **925 B**,`patch`/`test_patch`/`FAIL_TO_PASS`/`PASS_TO_PASS` 全 **ABSENT**;评测那份 5002 B 才完整 |
| 目标测试 | `unsquashfs` 直读 SIF 内 `tests/apps/tests.py` | 无 `test_clear_cache`;20 条 rollout 的工具调用记录 `touched_tests=0` |
| 用了新代码 | 读评测容器实际执行的 `eval.sh` | `git checkout <base_commit> tests/...` 后再 `git apply test_patch`;源码侧只应用**模型的** patch,金 patch 从未被应用 |

镜像 HEAD 与 `base_commit` 不一致曾是可疑点,已查明:`git show` 显示
`Author: SWE-bench <setup@swebench.config>` —— 那是造镜像时生成的 commit,代码树即
`base_commit` 状态。与"`unsquashfs` 读到镜像里缺那行修复"互相印证。

### 2.4 决定性反证

同一 instance、同一评测路径,12927 那条 `str_replace` 缩进错位的 patch 被判
`resolved=False`、`FAIL_TO_PASS failure=['test_clear_cache']`。评测**会**判错。

## 3. 这一步是有效训练步(前两跑不是)

| run | Advantages stats | 说明 |
|---|---|---|
| 12921 / 12927(单样本) | `std=nan` | 组内只有一条,无优势信号 |
| 12930(8/8 全对) | `std=0.0000` | 奖励全相等 → GRPO 组内归一化后 advantage 恒 0 |
| **12946** | `min=-0.7071 max=+0.7071 std=0.5000` | 组内有对有错,**非零梯度** |

±0.7071 正是 group size 2、一对一错时的归一化值 `(1-0.5)/0.7071`。

## 4. 复现

```bash
# 单题拿 reward=1
SWE_INSTANCE=django__django-17029 NUM_GENERATIONS=8 SWE_CONCURRENCY=4 \
  bash agent_run/scripts/phase4/launch_qwen3_swe2_tp2_e2e.sh

# 10 题合理性审计
SWE_DATASET=/workspace/agent_run/data/qwen3_swe2_smoke/django_mix10_swe2.jsonl \
NUM_PROMPTS=10 NUM_GENERATIONS=2 SWE_CONCURRENCY=10 \
  bash agent_run/scripts/phase4/launch_qwen3_swe2_tp2_e2e.sh

bash agent_run/handoff_qwen3_swe2_cold_start.sh --verify
```

harness:OpenHands(`sdevare-nv/nv-OpenHands` @ `5f01800`,`CodeActAgent`,`RUNTIME=local`),
这也是 NeMo-Gym 的默认值(`agent_framework: Literal["openhands","opencode"] = "openhands"`)。
评测侧是官方 `swebench.harness.run_local_evaluation`,与 agent 侧完全分离。

---

# 附录:被证伪的假设(来自已删除的 ROOT_CAUSE_qwen3_swe2_e2e.md)

原文件是 12921 时期的快照汇总,其"当前唯一阻断点:mamba"和"证据链 11/12"两节
在后续运行中已被推翻,留着会误导。下面这一节是它唯一不可替代的内容 —— 每条都是
花时间走过的死路,记下来是为了不让人重走。

## 清单

1. **venv 解释器不持久** — 错。`/root/.local/bin/python3.13` 烤在镜像里,宿主看着
   断链、容器内正常。据此加的「base_prefix 必须在 Lustre」断言本身是 bug,把两个
   安装都已成功的 prewarm(12895)判为失败,已删除。
2. **NEMO_GYM_SWE_NO_COPY 没传进容器** — 错,trace 显示其值就是 `1`。
3. **sandbox 冷启动慢 11 分钟 / 卡在 initialize_runtime 600s** — 错。
   `connect to runtime` 17~27s,`initialize_runtime` 40.73s 成功。
4. **PATH 被剥导致 jq 不可用** — 不准确。PATH 恢复后裸 jq 仍失败,真因是架构不符
   + PATH 顺序。
5. **eval 的 sleep busy loop 抢 CPU 拖慢 source** — 是真 bug,但非主因。
6. **`base_url=''` 是阻塞性缺陷** — 错。CodeActAgent 同样走 `NemoGymClient` 的
   server_name 路由(`codeact_agent.py:93`),空 base_url 是设计如此。
7. **引擎崩溃是 entry trace 造成的** — 错。12913 无 trace 同样崩溃。
   (trace 确有 `exec` 污染 session fd 的缺陷 —— 会打断 OpenHands 的 PS1 完成检测 ——
   但把引擎崩溃也归因于它是过度归因;两件事分开。)
8. **`container_formatter` 传字符串而非 list** — 不是问题,签名 `str | list[str]`。
9. **`DATASET_TYPE` 分派错误** — 不是问题,SWE-bench_Verified 本就走默认分支。
