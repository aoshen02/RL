# 根因:EngineCore KeyError —— async scheduling 让 PP=1 也进了"多并发 batch"路径

影响作业:12911(第一个真正把请求送到 vLLM 的作业)。

## 症状

第一个 `/v1/chat/completions` 到达的同一秒,EngineCore 致命退出:

    ERROR 08-13 09:17:39 [core.py:1233] EngineCore encountered a fatal error.
      core.py:605   step_with_batch_queue
      scheduler.py:1578  req_index = model_runner_output.req_id_to_index[req_id]
    KeyError: 'chatcmpl-9ffad060daaa175d-ab405602'

之后所有请求 500(`EngineDeadError`),Gym 重试 3 次全败,`prompt_tokens=0`,
GPU 全程 0%,作业以 exit 1 结束。

## 不是什么(已排除)

- **不是 abort 竞态**。0.25.1 在崩溃行的上方就有 `request is None or
  request.is_finished()` 的保护(注释明确写了 "aborted while the model is
  executing it (e.g., in pipeline parallelism or in async scheduling)")。
  已用 `git show v0.25.1:vllm/v1/core/sched/scheduler.py` 核对。
- **不是 `KeyError: None` 那一类**(vLLM #23223 / #25991):我们的 key 是真实的
  `chatcmpl-` id,并发也只有 1。
- **不是本地 vLLM 改动导致**。`/home/inf-aoshen/vllm` 里那 110 行未提交改动全是
  中文讲解注释,且容器用的是镜像内的 0.25.1,不走这个工作树。

## 是什么(源码验证)

`req_id` 在 `self.requests` 里且未完成,却不在 `model_runner_output.req_id_to_index`
中 —— 即 `scheduler_output.num_scheduled_tokens` 与 model runner 的输出不一致。
关键在于它是从 `step_with_batch_queue` 进来的。0.25.1:

    # vllm/v1/engine/core.py
    self.batch_queue_size = vllm_config.max_concurrent_batches
    if self.batch_queue_size > 1: ...            # 否则 batch_queue 为 None
    self.step if self.batch_queue is None else self.step_with_batch_queue

    # vllm/config/vllm.py::max_concurrent_batches
    pp_size = self.parallel_config.pipeline_parallel_size
    if self.scheduler_config.async_scheduling:
        if pp_size <= 1: return 2                # ← 我们命中这里
    return pp_size

我们 PP=1 却走了 `step_with_batch_queue`,**因此 async scheduling 是开着的**
(NeMo-RL 全仓库不设这个键,继承 vLLM 默认)。

## 与 NeMo-RL #2180 的关系,以及它的一处错误假设

NVIDIA-NeMo/RL#2180(in_flight_weight_updates 的竞态)给出的机制是:
vLLM `RayDistributedExecutor` 用 Ray compiled DAG(后台线程)派发 forward、
权重更新走主线程 `.remote()`;`step_with_batch_queue` **在 forward 完成前就返回**,
于是权重更新可以和在飞的 forward 重叠。该 issue 的结论是"PP=1 安全,因为
PP=1 强制 max_concurrent_batches=1、走阻塞 step()"。

**这个前提在 0.25.1 上不成立**:async scheduling 开启时 PP=1 也会得到 2。
我们这一跑正好构成它描述的危险组合:

- PP=1 + async scheduling → `max_concurrent_batches=2` → 非阻塞 `step_with_batch_queue`
- `load_format=dummy`(handoff 要求),所以启动时必须 refit 真权重
  (日志:`🔄 Refitting policy generation with actual model weights...`)
- `async_grpo(enabled=true, in_flight_weight_updates=true,
  recompute_kv_cache_after_weight_updates=false)`

## 处置

`policy.generation.vllm_kwargs.async_scheduling=false`。这是 NeMo-RL 的官方透传口子
(`vllm_worker.py:566` 把 `**vllm_kwargs` 直接展开进 `AsyncEngineArgs`),作用是把
`max_concurrent_batches` 恢复为 1、回到阻塞 `step()` —— 也就是恢复 NeMo-RL
in-flight refit 路径所假设的不变量。

它不改模型、token、reward、也不改 recipe 语义,但**确实是对 flagship 默认配置的一处
偏离**,并且有性能代价(async scheduling 本来用于消除 GPU 空隙)。因此在交付里
必须显式列为 deviation,不能当成"配好了"。

## 值得反馈上游

#2180 里"PP=1 is safe"的说法在 async scheduling 默认开启的版本上是错的:
判据应当是 `max_concurrent_batches > 1`,而不是 `pipeline_parallel_size > 1`。

## 更正:真正的根因是 mamba_cache_mode,与 async scheduling 无关(12919 证据)

`async_scheduling=false` 上线后(已用 config dump 与 `core.py:497 step` 双重确认生效),
KeyError 消失,但 EngineCore 仍然死,换成:

    multiproc_executor.py:391 get_response
    RuntimeError: Worker failed with error ''      ← 空消息

worker 侧的真实堆栈(`multiproc_executor.py:1004`):

    gpu_model_runner.py:4231  mamba_bufs = self._get_mamba_bufs()
    gpu_model_runner.py:1040  copy_funcs=self.model.get_mamba_state_copy_func()
    compilation/cuda_graph.py:220 __getattr__ → raise AttributeError   ← 裸抛,故上层为 ''

vLLM 0.25.1 的触发条件**只看开关,不检查模型是否含 mamba 层**:

    if self.cache_config.mamba_cache_mode == "align":
        mamba_bufs = self._get_mamba_bufs()

而 Qwen3-30B-A3B-Thinking 是纯 MoE transformer,没有 `get_mamba_state_copy_func`,
`CUDAGraphWrapper.__getattr__` 于是裸抛 AttributeError → 每个 forward 必死。
vLLM 默认值是 `mamba_cache_mode="none"`。

### 这个值是从哪来的:recipe 继承链

    grpo_qwen3_30ba3b_thinking_swe2.yaml
      → defaults: grpo_qwen3_30ba3b_instruct.yaml
          → defaults: grpo_workplace_assistant_nemotron_nano_v2_9b.yaml   (266-267 行)
                vllm_kwargs.mamba_ssm_cache_dtype: "float32"
                vllm_kwargs.mamba_cache_mode: align

Nemotron Nano v2 9B 是 mamba+attention 混合模型,这两个参数对它成立;Qwen3 的
recipe 挂在它上面却没有清掉 —— 这是 NeMo-RL 自带 example config 的继承缺陷,
不是我们的配置错误。

### 修复

    ++policy.generation.vllm_kwargs.mamba_cache_mode=none
    ++policy.generation.vllm_kwargs.mamba_ssm_cache_dtype=auto

即恢复 vLLM 默认。`async_scheduling=false` **已撤销** —— 它不是修复项,保留它只会
让最终交付多一处对 flagship 路径的无谓偏离。

### 前一节(async scheduling)的价值与更正

前一节关于 `max_concurrent_batches` 的源码结论仍然成立(PP=1 + async scheduling
→ 2,#2180 里"PP=1 is safe"在该版本上不成立,这条仍值得反馈上游);但把它当成
**本次崩溃的成因**是错的。它的实际作用是**诊断性的**:在 batch_queue 路径下
worker 死亡表现为 scheduler 里一个莫名的 `KeyError: chatcmpl-...`(输出永远回不来),
换成阻塞 `step()` 后 worker 的真实堆栈才冒上来。两次崩溃极可能是同一次 worker
死亡的两面。

### 方法论

"改了一处、症状变了"不等于"改对了"。这次是症状从 KeyError 变成 AttributeError,
如果当时就宣布修好、去追 KeyError 的旁支,会彻底走偏。判据始终应当是
**worker 侧的原始堆栈**,而不是上层看到的异常类型。
