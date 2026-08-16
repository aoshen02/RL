# 根因:SWE sandbox "600 秒超时" 是假象,真实故障在第 0 秒

影响作业:12881、12898、12900(同一故障)。12894/12895/12896 是另一类问题,见
`qwen3_swe2_tp2_e2e_20260813T075004Z/FAILURE_ANALYSIS.md`。

## 表面症状

OpenHands agent 日志:

    Failed to source /swe_util/instance_swe_entry.sh: **CmdOutputObservation
    (exit code=-1, pid=-1, ... [The command timed out after 600.0 seconds])**
    ----------[The above error occurred. Retrying... (attempt 1 of 3)]----------

600s × 3 次重试正好吃满 `swebench_agent_timeout: 1800`。据此很容易误判为
"sandbox 冷启动慢"或"/testbed 拷贝慢"。**这个判断是错的。**

同时 eval 容器日志被刷满:

    /container_scripts/eval_script.sh: line 1: /usr/bin/sleep: No such file or directory

## 真实原因(entry_trace.log 证据)

在 entry script 注入 `set -x` trace 后,一次运行即定性:

    [entry] 2026-08-13T08:27:18Z sourced; NEMO_GYM_SWE_NO_COPY=1 ...
    ++ [08:27:18] instance_swe_entry.sh:30: jq --arg INSTANCE_ID django__django-15037 ...
    ++ [08:27:18] instance_swe_entry.sh:30: item=
    ++ [08:27:18] instance_swe_entry.sh:32: [[ -z '' ]]
    ++ [08:27:18] instance_swe_entry.sh:33: echo 'No item found for the provided instance ID.'
    ++ [08:27:18] instance_swe_entry.sh:34: exit 1

全部发生在**同一秒**。因果链:

1. `_patch_openhands_swe_no_copy` 除了跳过 `/testbed` 拷贝,还顺带跳过了
   `source ~/.bashrc`,并把 `conda activate testbed` 换成裸 PATH/CONDA_PREFIX 导出。
2. 结果 conda 环境没有真正激活,`jq` 不在 PATH 上 → `item` 为空。
3. → 脚本走到 `exit 1`。该脚本是被 runtime **`source`** 的,`exit` 因此
   **终止调用方的 bash session**。
4. → OpenHands 在等一个已死 shell 的输出,直到 600s 硬超时。`exit code=-1,
   pid=-1` 正是"没有进程回话"的表现。
5. 同一原因让 eval 容器里 `sleep` 不可用,`until [ -f X ]; do sleep 5; done`
   退化成无节流 busy loop,既刷日志又抢 CPU。

数据侧无问题,已核对:挂载的 `/root/dataset/data.jsonl` 中
`instance_id == django__django-15037`,`run_infer.py` 也正确地写成
`json.dump([instance])` 数组。`NEMO_GYM_SWE_NO_COPY=1` 也确实传进了容器
(trace 首行),此前"env 丢失"的怀疑同样被推翻。

## 修复

1. `_patch_openhands_swe_no_copy` 先对 `instance_swe_entry.sh` 与
   `run_infer.py` 执行 `git checkout --` 还原,再只做一件事:把
   `cp -al /testbed` 换成 `ln -s /testbed`。`source ~/.bashrc` 与
   `conda activate testbed` 全部保持上游原样 —— 后者正是执行
   `$CONDA_PREFIX/etc/conda/activate.d/*.sh` 的地方。
2. `_get_command_sleep_until_predictions_file` 加多级回退
   (`sleep` → `/bin/sleep` → `/usr/bin/sleep` → `python3 time.sleep`),
   全失败时打一次告警而非静默空转。
3. entry script 保留 trace 注入(幂等,`NEMO_GYM_ENTRY_TRACE` 标记),
   输出到 `/trajectories_mount/entry_trace.log`,该路径 bind 自
   `persistent_dir`,运行中即可从宿主机读取。

## 附带结论

原先记录的审计发现"no-copy 路径跳过 activate.d,不等价于完整 benchmark"不再是
待办限制 —— 该路径已不再跳过任何环境初始化,因此对完整 benchmark 也是等价的。
唯一差异是 workspace 为 `/testbed` 的软链而非副本,依赖
`apptainer --writable-tmpfs` 提供可写覆盖层。

## 方法论教训

`test -x`、"进程还在跑"、"日志没报错"都不是证据。这次三轮误判(解释器不持久 →
env 丢失 → sandbox 慢)全部源于对不可见区段的推测;把子进程 stdout 重定向到
宿主可见路径并加时间戳 trace 之后,一轮就定性。子进程输出被吞掉的地方就是
必须先补可观测性的地方。

## 修正:上一节的根因只对了一半(12908 证据)

12908 的 entry trace 证明 conda 修复生效了 —— `source ~/.bashrc` 执行、PATH 含
`/opt/miniconda3/envs/testbed/bin`、`activate.d` 跑完、`SWE_INSTANCE_ID` 正确、
`swe-bench-instance.json` 存在且首行就含 `"instance_id": "django__django-15037"`。
但 `item=` **依然为空**,`exit 1` 照旧。trace 里多打的那行探针给出了真凶:

    bash: /usr/local/bin/jq: cannot execute binary file: Exec format error

即 jq 是 **x86_64 二进制,跑在 aarch64(GB200)上**。所以"conda 没激活导致 jq
不在 PATH"只是次要因素;jq 即便在 PATH 上也无法执行。

### 上游 bug

`responses_api_agents/swe_agents/setup_scripts/openhands.sh` 对 miniforge 是
`uname -m` 感知的,但第 55 行把 jq 的下载地址**硬编码成 `jq-linux-amd64`**。
`app.py` 随后 `cp /openhands_setup/miniforge3/bin/jq /usr/local/bin/jq`,于是这个
x86 二进制被塞进 ARM 容器,并覆盖镜像自带的 jq。宿主实测:

    swe_openhands_setup/miniforge3/bin/jq: ELF 64-bit LSB executable, x86-64
    uname -m: aarch64

### 修复(双层)

1. **主修**:`openhands.sh` 按 `uname -m` 映射 `amd64`/`arm64`,并把存在性判断
   改成"存在**且能执行**"(`jq --version`),否则重下;结尾的 `jq --version`
   去掉 `|| true`,让错架构在 setup 阶段就炸而不是拖到 sandbox 里。
   由于 setup 在 `.venv` 已存在时会提前返回、不会重跑,已就地把磁盘上那个 x86
   jq 换成 arm64(`jq-1.8.1`,`file` 与 `--version` 均已核对),旧的留为
   `jq.amd64.bak`。
2. **兜底**:entry script 注入 `_ng_select_instance` / `_ng_workspace_name`。
   两者先探测真 jq(`jq --version`),可用就走上游原路径,不可用才用容器内
   python 复现这两条查询。这样 x86 主机上行为逐字不变。

### 验证

- 用 `ast` 抽出 `_patch_openhands_swe_no_copy` 真身执行:shim 注入成功、
  `ln -s /testbed` 保留、`bash -n` 通过、二次执行结果逐字节相同(幂等)。
  残留的裸 jq 调用只存在于 `_ng_jq_works` 判真分支内。
- 两条分支对同一份 instance JSON 输出一致(真 jq 与"假 jq 触发 Exec format
  error"两种 PATH 下均得 `workspace=django__django__4.1`,查不到时均为空串)。

### 方法论(再次印证)

这轮之所以还能再错一次,是因为上一轮把"conda 激活了"当成了"jq 可用了"。
真正让它收敛的仍是同一件事:把不可见区段的实际输出落到宿主可读路径 ——
这次是 trace 里那一行 `jq ids present:` 探针,而不是任何推理。

## 第三个原因:我加的 trace 本身制造了同一个 600s 挂起(12909 证据)

12909 里 jq 修复生效、entry 判定全绿(`exit1=0 symlink=1 completed=1`,08:59:10
一秒内跑完),但 OpenHands 依然卡住。py-spy 直接给出栈:

    initialize_runtime (run_infer.py:1070)   ← source instance_swe_entry.sh 那条 action
    send_action_for_execution → httpx 等响应

action server 侧同时停在 `bash.py:934` 的轮询里(等 pane 出现 PS1 标记),
`TmuxMemoryMonitor` 在 `bash.py:126` 的正常 sleep 分支(不是 kill 分支,已排除)。
宿主进程树显示 pane shell(`bash -`)**没有任何子进程** —— 命令确实结束了。

### 真正的机制

trace prelude 的第一行是:

    exec 9>>/trajectories_mount/entry_trace.log 2>/dev/null || exec 9>/dev/null

`exec` **不带命令**时,它的重定向会**永久作用于当前 shell**。所以 `2>/dev/null`
把 shell 的 stderr 永久丢弃了 —— 而 **bash 的提示符(PS1)写在 stderr 上**。
这个脚本是被 `source` 进 agent 那个长驻 bash session 的,于是该 session 之后
永远不再输出提示符;OpenHands 靠在 tmux pane 里找 `###PS1END###` 判断命令结束
(`bash.py:862`,`CMD_OUTPUT_METADATA_PS1_REGEX`),于是每条 blocking action 都
只能等到 600s 硬超时,再重试 3 次吃满 1800s。

宿主上一行即可复现:

    exec 9>>/tmp/x.log 2>/dev/null; echo X >&2   # X 不会出现

这一行恰恰是我为"防止 exec 失败导致挂起"加的兜底,结果它以另一种方式造成了
同样的挂起 —— 而且症状与真正的 jq 故障**逐字相同**,这正是它此前一直藏得住的原因。

### 修复

删除整块 trace 注入(它的使命已经完成:x86 jq 就是它找出来的)。保留的只有:
jq shim、`ln -s /testbed`、以及两行普通 `echo ... 2>/dev/null >>/trajectories_mount/entry.log`
—— 有命令的重定向只作用于该命令,绝不改动 shell。`2>/dev/null` 前置,避免
`>>` 打开失败时把报错写进 agent 的 pane。

已验证:补丁后脚本无任何裸 `exec`、无 `BASH_XTRACEFD`、无 `set -x`;
`source` prelude 之后 stderr 仍然存活、pane 无噪音;`bash -n` 通过;幂等。

### 方法论

这轮的决定性工具是 py-spy(`ptrace_scope=0`,容器进程在宿主上就是本人 uid),
它把"卡在哪"从推测变成了一行栈帧。前面三次误判都是因为只能看到"没有输出";
一旦能读到 Python 栈和宿主进程树,"命令已结束但提示符没出现"这个结论是直接读出来的。
另外记一笔:同一个症状(600s 超时)在这条链上至少有三种独立成因,
所以"改了一处、症状还在"绝不等于"上一处没修对"。
