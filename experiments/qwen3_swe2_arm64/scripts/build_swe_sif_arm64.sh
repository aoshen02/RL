#!/usr/bin/env bash
# 为一个 SWE-bench instance 造 ARM64 SIF。
#
# 为什么可以零定制:已实测(unsquashfs -l)现有的 django-15037 SIF **就是上游 docker
# 镜像的直接转换** —— 里面只有 /opt/miniconda3、/testbed、/root/setup_{env,repo}.sh,
# 没有 /swe_util,也没有 /root/dataset。那两个是运行时由 Gym swe_agents 注入的
# (app.py:3277 `--mount type=bind,...,dst=/root/dataset/data.jsonl`)。
# 所以造镜像只需一条 apptainer build,不需要复刻任何目录结构。
#
# 宿主机没有 apptainer(只在 enroot 容器内),docker.sock 也没权限,所以必须走 Slurm。
#
# 用法:  build_swe_sif_arm64.sh <instance_id>        # e.g. django__django-17029
set -euo pipefail

IID="${1:?usage: $0 <instance_id>   e.g. django__django-17029}"
NMU=/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu
AGENT_RUN="$NMU/agent_run"
IMAGE=/mnt/lustre01/users/inf-aoshen/enroot/containers/nemo-rl-vllm-latest.sqsh
SIF_DIR=/mnt/lustre01/users/inf-aoshen/swe-sif-arm64

# 上游仓库名把 `__` 写成 `_1776_`,输出文件名沿用现有 SIF 的 `swebench_` 前缀,
# 因为 app.py 用 `*{instance_id}*.sif` 做模糊匹配,前缀无所谓但保持一致便于对照。
REPO_ID="${IID/__/_1776_}"
REPO="swebench/sweb.eval.arm64.${REPO_ID}"
OUT="$SIF_DIR/swebench_sweb.eval.arm64.${REPO_ID}.sif"

if [[ -s "$OUT" ]]; then
  echo "already built: $OUT ($(du -m "$OUT" | cut -f1) MB)"; exit 0
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$AGENT_RUN/results/build_sif_${REPO_ID}_${TS}"
mkdir -p "$RUN_DIR" "$SIF_DIR"

cat > "$RUN_DIR/build.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
echo "[build] \$(date -u +%FT%TZ) node=\$(hostname) arch=\$(uname -m)"
command -v apptainer || command -v singularity || { echo "FATAL: no apptainer in image"; exit 3; }
APP="\$(command -v apptainer || command -v singularity)"
"\$APP" --version

# cache 放 Lustre 可复用;tmpdir 必须留在节点本地 —— apptainer build 要在 tmpdir 上
# 铺 overlay/sandbox,而 overlayfs 不接受网络文件系统做 lowerdir(enroot 上踩过)。
export APPTAINER_CACHEDIR=/mnt/lustre01/users/inf-aoshen/apptainer_cache
export SINGULARITY_CACHEDIR="\$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR=/tmp/apptainer_build_\$\$
export SINGULARITY_TMPDIR="\$APPTAINER_TMPDIR"
mkdir -p "\$APPTAINER_CACHEDIR" "\$APPTAINER_TMPDIR"
df -h /tmp "\$APPTAINER_CACHEDIR" | sed 's/^/[df] /'
trap 'rm -rf "\$APPTAINER_TMPDIR"' EXIT

# 先只查 manifest:拉 1GB 之后才发现 tag 不存在很浪费。
echo "[build] \$(date -u +%FT%TZ) probing docker://$REPO"
TMP_SIF="\$APPTAINER_TMPDIR/out.sif"
time "\$APP" build --arch arm64 "\$TMP_SIF" "docker://$REPO:latest"

echo "[build] \$(date -u +%FT%TZ) verifying built image"
"\$APP" exec --containall "\$TMP_SIF" bash -lc '
  set -e
  echo "arch=\$(uname -m)"
  test -d /testbed && echo "testbed=OK"
  test -x /opt/miniconda3/bin/conda && echo "conda=OK"
  git -C /testbed rev-parse HEAD
' 2>&1 | sed 's/^/[verify] /'

# 原子落盘:半个 SIF 留在最终路径上会被 app.py 的模糊匹配挑中。
mv "\$TMP_SIF" "$OUT.partial"
mv "$OUT.partial" "$OUT"
ls -l --block-size=M "$OUT"
echo "[build] \$(date -u +%FT%TZ) done -> $OUT"
EOS
chmod +x "$RUN_DIR/build.sh"

JOB=$(sbatch --parsable \
  --job-name="build_sif_${REPO_ID}" \
  --account=inferact \
  --partition=batch \
  --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --mem=48G \
  --time=01:00:00 \
  --output="$RUN_DIR/slurm-%j.log" \
  --container-image="$IMAGE" \
  --container-mounts="$AGENT_RUN:/workspace/agent_run,/mnt/lustre01:/mnt/lustre01" \
  --container-remap-root \
  --wrap="bash /workspace/agent_run/results/$(basename "$RUN_DIR")/build.sh")

echo "instance   = $IID"
echo "docker     = docker://$REPO:latest"
echo "target sif = $OUT"
echo "job        = $JOB"
echo "log        = $RUN_DIR/slurm-$JOB.log"
