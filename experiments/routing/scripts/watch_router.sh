#!/usr/bin/env bash
# 实时看 router 把多轮 session 路由成了什么样。
#
#   bash watch_router.sh <run_dir> [间隔秒]
#
# 每个采样点一行 per-worker：
#   hit%   本次间隔内新增的 prefix cache 命中 token / 查询 token
#          —— session-aware 路由生不生效就看这个，同一 session 的第 N 轮
#             被送回同一 worker 才可能命中前 N-1 轮的 KV
#   req    累计完成请求数（看负载是否倾斜）
#   run/wt 当前 running / waiting，看有没有排队
set -u

RUN_DIR="${1:?用法: watch_router.sh <run_dir> [间隔秒]}"
INTERVAL="${2:-15}"
# 调用方可能在引擎起来之前就把采样器拉起来了，所以这里自己等，而不是直接退出
for _ in $(seq 1 120); do
  ROUTER=$(grep -oE "router_url=http://[0-9.]+:[0-9]+" "$RUN_DIR/nemo_rl.log" 2>/dev/null |
    head -1 | cut -d= -f2-)
  [ -n "${ROUTER:-}" ] && break
  # 引擎已就位却始终没有 router_url ⇒ ROUTER_POLICY=none，别再等了
  grep -q "Reserved [0-9]* vLLM server URLs" "$RUN_DIR/nemo_rl.log" 2>/dev/null && break
  sleep 10
done
# ROUTER_POLICY=none 没有 router，但引擎指标一样要采：
# 引擎 URL 直接取 nemo_rl.log 里的 Reserved 行，三种策略统一走这条路兜底
engines_from_log() {
  grep -m1 -oE "Reserved [0-9]+ vLLM server URLs: \[[^]]*\]" "$RUN_DIR/nemo_rl.log" 2>/dev/null |
    grep -oE "http://[0-9.]+:[0-9]+"
}

# 一律用 Reserved 行,不问 router 要列表:注册是逐个阻塞进行的,问 router 会拿到
# 注册到一半的部分列表并就此定死(13233 只监控到 4/28 台,13234 只有 7/28)。
[ -n "${ROUTER:-}" ] || ROUTER="(none)"
for _ in $(seq 1 120); do
  WORKERS=$(engines_from_log)
  [ -n "${WORKERS:-}" ] && break
  sleep 10
done
[ -n "${WORKERS:-}" ] || { echo "拿不到 worker 列表（router=$ROUTER）"; exit 0; }

echo "router $ROUTER"
echo "worker $(echo $WORKERS | tr '\n' ' ')"

declare -A PQ PH
metric() { grep -m1 "^vllm:$1{" <<<"$2" | awk '{print $NF}'; }

while :; do
  # epoch 一起打，nonfinite dump 里的 ts 才能对齐到具体采样点
  line="$(date -u +%H:%M:%S) t=$(date +%s)"
  tq=0; th=0
  for w in $WORKERS; do
    m=$(curl -s --max-time 5 "$w/metrics" 2>/dev/null)
    [ -n "$m" ] || { line="$line | $(basename $w) --"; continue; }
    q=$(metric prefix_cache_queries_total "$m"); h=$(metric prefix_cache_hits_total "$m")
    req=$(grep "^vllm:request_success_total{" <<<"$m" | awk '{s+=$NF} END{print s+0}')
    run=$(metric num_requests_running "$m"); wt=$(metric num_requests_waiting "$m")
    # 抢占次数和 KV 占用：判断「KV 压力」这个假设唯一的直接证据，之前靠全局容量
    # 算术推断，那是平均值，掩盖了单个引擎的真实水位
    pre=$(grep -m1 "^vllm:num_preemptions_total{" <<<"$m" | awk '{print $NF}')
    # 指标名跨版本变过:旧 gpu_cache_usage_perc,新 kv_cache_usage_perc,都试
    kv=$(grep -m1 -E "^vllm:(gpu|kv)_cache_usage_perc\{" <<<"$m" | awk '{print $NF}')
    dq=$(awk -v a="${q:-0}" -v b="${PQ[$w]:-0}" 'BEGIN{print a-b}')
    dh=$(awk -v a="${h:-0}" -v b="${PH[$w]:-0}" 'BEGIN{print a-b}')
    PQ[$w]=${q:-0}; PH[$w]=${h:-0}
    tq=$(awk -v a=$tq -v b=$dq 'BEGIN{print a+b}'); th=$(awk -v a=$th -v b=$dh 'BEGIN{print a+b}')
    pct=$(awk -v h=$dh -v q=$dq 'BEGIN{if(q>0) printf "%5.1f%%",100*h/q; else printf "    -"}')
    kvp=$(awk -v v="${kv:-0}" 'BEGIN{printf "%2.0f%%",100*v}')
    line="$line | ${w##*:} $pct req=${req%.*} ${run%.*}/${wt%.*} kv=$kvp pre=${pre%.*}"
  done
  tot=$(awk -v h=$th -v q=$tq 'BEGIN{if(q>0) printf "%.1f%%",100*h/q; else printf "-"}')
  echo "$line || 合计 $tot"
  sleep "$INTERVAL"
done
