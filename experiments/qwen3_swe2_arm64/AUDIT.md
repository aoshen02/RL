# 独立审计报告 —— 作业 12921(qwen3_swe2_tp2_e2e_20260813T094539Z)

满足 handoff 的 AUDIT REQUIREMENT:对 venv 路径推导、包/导入兼容、Apptainer 内
环境传递、no-copy 沙箱语义、缓存/锁竞态、以及"最终这一跑是否真的是 flagship
NeMo-RL/Gym 路径"六个方面做独立只读审计,并把结论留在 run 结果目录。

三份审计各自独立进行(只读、要求 file:line 证据)。下面按"我的裁决"分三类:
**已确认**、**被实证推翻**、**开放风险**。裁决依据只用当前盘上的文件与日志。

---

## A. 已确认(与实证一致)

1. **venv 路径推导正确,且是结构性正确而非碰巧。**
   `setup_command.py:106-110` 的 `venv_path = Path(root, *dir_path.parts[-2:], ".venv")`
   中 `parts[-2:]` 恒等于配置键 `responses_api_models/vllm_model` 与
   `responses_api_agents/swe_agents`(`env.py:80-90,205`),与 prewarm 创建的、
   run 脚本校验的两条路径逐字一致。train/val 两个 SWE server 因 `parts[-2:]`
   相同而共用一个 venv(`swebench_openhands_training.yaml:5,57`)。

2. **`NEMO_GYM_VENV_DIR` 的镜像覆盖确实存在**(`docker/Dockerfile:123`),所以 run
   脚本用 `QWEN3_SWE2_GYM_VENV_DIR` 绕过、再无条件重设的做法是必要的,不是多余。

3. **jq 修复正确且自愈**:`openhands.sh:53-57` 按 `uname -m` 派发;`:62` 在既有
   二进制 `--version` 失败时重下;`:70` 去掉 `|| true` 后错架构在 setup 阶段即失败。
   实测 `miniforge3/bin/jq` = ARM aarch64。审计另外指出我遗留的 `jq.amd64.bak`
   (x86)还在被 bind-mount 且 PATH 前置的目录里 —— **已删除**,现该目录无任何
   非 aarch64 ELF。

4. **mamba 根因独立确认**:`grpo_workplace_assistant_nemotron_nano_v2_9b.yaml:266-267`
   设 `mamba_ssm_cache_dtype: float32` / `mamba_cache_mode: align`,经 defaults 合并
   进入 Qwen3 recipe;而 vLLM 自己的默认值正是 `cache.py:135` `"auto"` 与
   `cache.py:139` `"none"`。因此我的两个覆盖是**恢复原厂行为**,不是调优偏离。

5. **`async_engine=true` 是真正的 bug fix,不是 handoff 指定项(修正我先前的表述)。**
   `grpo_qwen3_30ba3b_instruct.yaml:67` 把它覆盖成 `false`,而 NeMo-Gym 在
   `grpo.py:2078` 硬要求 async。不加这个覆盖 flagship recipe 起不来。
   → 建议在 `grpo_qwen3_30ba3b_thinking_swe2.yaml` 里直接写 `async_engine: true`。

6. **这一跑确实走了真实 NeMo-RL 训练路径,不是 vLLM-only 捷径。**
   `slurm-12921.log:449` Megatron 载入 checkpoint、`:503-504` refit 真权重、
   `:1000-1029` ReplayBuffer 取样、`:1029` 起真实反向传播、`:1050-1056` 训练后
   refit 协调。`load_format=dummy` 不是 CLI 覆盖,来自配置默认,且由 refit 补齐 —— 
   这正是标准 NeMo-RL 路径。

7. **conda `activate.d` 在 no-copy 路径下完整保留**(entry script `:28` `source ~/.bashrc`、
   `:72-75` `conda activate testbed` 均未改动;`run_infer.py` 由 `app.py:1581-1585`
   还原为上游)。因此 no-copy 不再跳过任何环境初始化。

8. **jq shim 行为保持**:先探测真 jq、可用即走上游逐字路径,不可用才用 python 复现
   两条查询(含"查不到即空串 → `exit 1`"的语义)。审计指出本次 aarch64 上实际
   走的是 python 回退分支,真 jq 分支未被本跑覆盖。

9. **`UV_LOCK_TIMEOUT` 确实是有效开关**且能传到各 server(`nemo_gym.py:706-712`
   的 `runtime_env env_vars`、`setup_command.py:181` 的 `environ.copy()`)。

10. **进入容器的二进制均为 aarch64 且 glibc-2.17 干净**(`tmux`/`node`/`python3.12`/`git`
    最高版本化符号为 `GLIBC_2.17`),无"glibc 比镜像新"的隐患。

---

## B. 被实证推翻的审计结论(记录以免后人误信)

11. 审计三怀疑"`no_copy=1` 分支可能从未真正执行过"(它只查到一份早于该 flag 的
    旧日志)。**推翻**:本 run 目录对应沙箱的 `entry.log` 首行为
    `[entry] 2026-08-13T09:52:08Z start id=django__django-15037 no_copy=1`,
    末行 `completed workspace=django__django__4.1`。已执行。

12. 审计三称 trace 行必然记成 `id=UNSET`,因为它在 `source ~/.bashrc` 之前。
    **推翻**:实测记录的是 `id=django__django-15037`。原因是 OpenHands 的长驻
    bash session 在此之前已经 source 过 `~/.bashrc`,变量已在环境中。

13. 审计一将"三个 server 并发 `uv pip install` 到同一目录"标为 blocker。
    **降级为条件性风险**:在 `skip_venv_if_present: true` + venv 已预热时
    (`setup_command.py:116-117`)各 server 只 source activate,无竞态。本跑
    `EVIDENCE.txt` 的 "cold-start: no install = YES" 即为实证。但该结论**依赖
    run 脚本里那个 preflight**,见 14。

---

## C. 开放风险(未修,按影响排序)

14. **run 脚本的 Gym venv preflight 是承重结构,不能删。**
    `env.py:205-217` 用 `Popen` 立即并发拉起三个 server,`train`/`val` 两个 SWE
    server 指向同一 venv 目录且**无任何锁**;venv 一旦缺失即重现作业 12876 的
    并发安装死锁。建议进一步加强:除 `import ray, openai` 外,再断言
    `import nemo_gym` 可用、且三者版本与父 venv 一致(Gym 在
    `global_config.py:691-701` 是按父 venv 版本生成子依赖的,而 prewarm 里是硬编码)。

15. **`--writable-tmpfs` 的 overlay 容量上限无法在本节点确认**(该节点无
    `/etc/apptainer/apptainer.conf`、PATH 上无 `apptainer`)。no-copy 下
    `git repack -ad`/`git gc`(`run_infer.py:869-878`)与测试产物都写进这层内存
    overlay;`app.py:3214-3219` 已因同样的上限把 Gradle home 移到
    `/trajectories_mount`,说明这个上限确实会咬人。本跑仅 4 轮、未执行测试,
    **不足以支撑满量 benchmark 的结论**。建议改用显式 sized overlay 或
    per-instance scratch 目录。

16. **`UV_CACHE_DIR` 靠两环链条恰好正确。** `global_config.py:709-711` 无条件覆写
    该变量,取值来自 `uv cache dir`,而这条路径仅在 `NRL_CONTAINER` 存在时成立
    (镜像里有,`Dockerfile:55`)。任一环断掉就会退化成
    `<Gym repo>/cache/uv`,把数 GB 缓存写进挂载的 host git worktree。
    建议显式加 `env.nemo_gym.uv_cache_dir=<lustre 路径>` 覆盖,不再依赖探测。

17. **jq shim 只覆盖 `instance_swe_entry.sh`,且只在 `NO_COPY=1` 时安装**
    (`app.py:1564,1567-1570`)。其余五个按数据集选择的 entry 脚本
    (`_live/_r2e/_rebench/_nv_internal/_swe_ext`)仍裸调 jq,在 aarch64 上会重现
    同一个"sourced 脚本 `exit 1` → 600s 假超时"。对本次数据集无影响。
    建议把 shim 安装移出 `NO_COPY` 判据并应用到全部 `instance_swe_entry*.sh`。

18. **跨作业共享状态,无跨作业锁**:
    - `swe_openhands_setup` 位于挂载的 worktree、被同一 checkout 的所有作业共享;
      B 作业启动时会 resync 并重写 entry script,而 A 作业的 agent 正在用这些文件
      (`app.py:1517-1537,1718-1719`)。
    - `NRL_MEGATRON_CHECKPOINT_DIR` 的 HF→Megatron 转换无锁,"已存在"判据
      (`setup.py:547-549`)在写一半时即为真 → 并发作业可能载入残缺 checkpoint。
      本跑是命中已暖好的缓存(`slurm-12921.log` 直接 loading,不是转换),无碍。
    - `VLLM_CACHE_ROOT` 的隔离只到 DP rank,不到作业。
    - Gym 的 `file_lock`(`app.py:310-340`)持锁期间不刷新 mtime,超过 1 小时的
      setup 会被兄弟进程判定为 stale 并 `rmtree`。
    结论:**当前"一次只跑一个 e2e 作业"的用法是安全的;并发跑需要先解决这些。**

19. **`MEGATRON_CONFIG_LOCK_DIR` 在本 stack 里没有消费者** —— 全仓库仅
    `examples/nemo_gym/nemotron-3-super/super_launch.sh` 使用,无 Python 读取;
    `Megatron-Bridge` 子模块在本 worktree 为空,无法确认镜像内是否有消费者。
    我按参考脚本 `grpo_workplace_assistant.sh` 配了它,**不应被当作已生效的
    HF-config 锁**。

20. **`HOME` 未通过 `--env` 传入**,而流程依赖 `~/.bashrc` 可写
    (`run_infer.py:1006-1013` 追加并断言 exit_code==0)。本跑成功说明容器内
    以 uid 0 运行、`/root` 存在;但这是隐式依赖。建议显式 `--env HOME=/root`。

21. **代理/TLS/index 环境变量未转发**(`--cleanenv` 会丢掉
    `HTTP(S)_PROXY`/`SSL_CERT_FILE`/`PIP_INDEX_URL`),容器内的 pip 安装步骤
    (`app.py:1830-1838` 硬编码 `--index-url https://pypi.org/simple`)在有代理的
    集群上会失败。本集群未受影响。

22. **`{swebench_setup_dir}/uv/bin` 这个 PATH 目录不存在**(`app.py:429,499,577`;
    `swebench.sh:15` 让 uv 平铺在 `uv/` 下)。当前无害只因 eval 用绝对路径调
    `SWE-bench/venv/bin/python`;一旦 eval 命令里新增 `uv` 调用就会
    `command not found`。

23. **OpenHands 的 `.venv` 不可迁移**:`pyvenv.cfg` 与 `bin/python` 均指向绝对路径
    `/opt/nemo-rl/3rdparty/.../miniforge3/bin/python3.12`,仅因 `app.py:3339-3340`
    把 miniforge3 双重挂载(容器路径 + 宿主原路径)才解析成功。且其 fast-path 判据
    用 `Path(...).exists()` 会跟随符号链接 —— 与 run 脚本刻意"执行解释器"而非
    `test -x` 的教训同类。建议该 fast-path 也改为执行式检查。

---

## D. 作用域声明(交付时必须一并说明)

24. **`enable_flashinfer_autotune=false` 是真实偏离。** recipe 链中无此项;它省掉的
    正是 flagship 会付的 autotune 启动成本,因此 `timeline.tsv` 的数字
    **不可作为 flagship 冷启动基准**。(该项由用户显式要求加入。)

25. **`grpo.max_rollout_turns` 解析为 1**(继承自 Nemotron recipe),即每样本一次
    环境 rollout;`reward=0.0`、`advantage std=0`,意味着这一步优化实际是零更新。
    链路完整性成立,但"训练有效性"未被本跑验证。

26. **`agent_max_turns=200` 与 `colocated.*` 的覆盖与 recipe 原值相同**,是冗余重述
    而非偏离;实际只用了 4 轮,200 轮路径(及其 overlay 增长)未被覆盖。

27. **本跑的 host 特异项**:`NEMO_GYM_SWE_NO_COPY=1` 与 aarch64 jq 回退分支
    **不属于 flagship x86 路径**。

28. **handoff 里 before-rerun 的 `test -x <venv>/bin/python` 在宿主上必然 FAIL**
    (解释器是指向容器内 `/root/.local/bin/python3.13` 的符号链接),见
    `handoff_check.txt`。有效判据是 run 脚本内**执行**解释器的 preflight,
    本跑为 YES。

---

## E. 三份审计各自声明的"无法验证"项(汇总)

- Apptainer overlay 上限、`apptainer.conf`、`--no-mount bind-paths` 与站点配置的交互。
- 容器内实际 uid / `HOME`,以及 SIF 自带 `/usr/bin/jq` 是否确为 x86(代码注释如此声称)。
- 镜像内 `ray`/`openai`/`python`/`uv` 的确切版本,以及镜像内是否存在设置
  `cache-dir` 的 `uv.toml`(会改变第 16 条的结论)。
- `MEGATRON_CONFIG_LOCK_DIR` 的消费者(子模块目录为空)。
- 真 jq 分支、多轮/执行测试/多实例的 no-copy 行为、以及 flagship 规模
  (16 节点、concurrency 768)下的任何行为。

---

## F. 我刻意未做的修改,以及原因

第 14/16/17/20/21/22/23 条都是有价值的加固,但**本次没有改**:运行脚本与
`EVIDENCE.txt`/`trajectory_step1.*` 是一一对应的证据,现在改脚本会让"已验证的
那一跑"与脚本不再一致。这些应作为下一轮的独立改动,改完各自重新验证 —— 
本轮的教训正是"改了一处、症状变了"不等于"改对了"。

唯一即时执行的是删除 `jq.amd64.bak`:它是我遗留的 x86 二进制、位于会被挂进容器
并前置到 PATH 的目录,且不影响已验证的运行配置。

## 校验器自检 #2(负向测试,2026-08-13,复问 "verify 逻辑真的启动了吗")

方法:把 run 目录整份复制到 scratchpad,改坏三处信号后跑 verify:
`Step 1/1`→`Step X/X`、`Wait iteration N: buffer_size=`→`bufsizeX=`、`exit_code=0`→`exit_code=1`。

结果(第一次):item 10、11 正确变 NO,判定 FAIL / exit 1 —— 失败路径确实 fire。
**但 item 12 在 exit_code=1 时仍报 YES** —— 原实现只 grep `exit_code=` 是否存在,
不校验其值。这是一个真实漏洞:一个失败退出的 run 会在这一项上过关。

修复:`collect_evidence_qwen3_swe2.sh` item 12 改为解析值并要求 `== 0`。
修复后复测:负向 → item 12 NO,FAIL/exit 1;真实 run → 12 项全 YES,PASS/exit 0。

### 仍存在的弱点(未改,如实记录)
item 3/4/5/8 的证据源在共享的 Gym results 目录下,靠 `since` 时间过滤归属本次 run;
若同一时刻有另一个 job 写入同一目录,归属可能串。单节点单 job 的场景下不触发。

## Slurm 分区是 OverSubscribe=EXCLUSIVE(2026-08-13,造 10 个 SIF 时发现)

`scontrol show partition batch` → `OverSubscribe=EXCLUSIVE`。每个作业独占整节点,
不论申请多少 CPU/内存:一个 `--cpus-per-task=16 --mem=48G` 的作业实际拿到
`NumCPUs=144`、`CPUAlloc=144`。

后果:在这个集群上**提交 N 个小作业 = N 倍墙钟**。造多个 SIF 应该塞进一个 sbatch
里并行(节点 144 CPU / 914 GB),而不是提交 N 个 job。

一次错误诊断的记录:我先以为串行是因为 `DefMemPerNode=UNLIMITED` 让每个作业抓走
整节点 914 GB 内存。加 `--mem=48G` 后内存记账确实修正了(AllocMem 914192→49152),
但**串行没变** —— 说明内存不是原因。这是"改动看起来合理、指标也变了、但没解决问题"
的典型:必须回去看 pending 作业的 Reason 和分区策略才能定因。
