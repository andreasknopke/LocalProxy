#!/usr/bin/env python3
"""
Deploy TurboQuant-enhanced vLLM Docker image on DGX Spark.
Extends avarok/dgx-vllm-nvfp4-kernel:v23 with TurboQuant (3-bit keys, 2-bit values).

Usage:
    python deploy_turboquant_docker.py

What it does:
    1. Copies Dockerfile.turboquant & turboquant-entrypoint.py to the Spark
    2. Pulls base image (if not already present)
    3. Builds new image: avarok/dgx-vllm-nvfp4-kernel:turboquant
    4. Updates run_qwen.sh to use the new image with TurboQuant args
    5. Updates systemd and restarts the service
"""
import paramiko, time, os

HOST = "192.168.188.185"
USER = "owc"
PASS = "OpenSourceRulez!"
SERVICE_NAME = "qwen-coder"
APP_DIR = "/home/owc/qwen-coder-vllm"
BASE_IMAGE = "avarok/dgx-vllm-nvfp4-kernel:v23"
NEW_TAG = "avarok/dgx-vllm-nvfp4-kernel:turboquant"
MODEL = "GadflyII/Qwen3-Coder-Next-NVFP4"
PORT = 8000

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS, look_for_keys=False)
print(f"Connected to {HOST}!")

def ssh(cmd, t=300):
    print(f"  $ {cmd[:150]}", flush=True)
    i, o, e = c.exec_command(cmd, timeout=t, get_pty=True)
    i.write(PASS + "\n")
    i.flush()
    o.channel.recv_exit_status()
    out = o.read().decode("utf-8", errors="replace")
    err = e.read().decode("utf-8", errors="replace")
    for line in (out + "\n" + err).strip().split("\n")[-20:]:
        if line.strip():
            print(f"    {line}", flush=True)
    return out, err

def sudo(cmd, t=300):
    return ssh(f"echo '{PASS}' | sudo -S bash -c '{cmd}'", t)

def sftp_put(local_name, remote_name):
    local_path = os.path.join(LOCAL_DIR, local_name)
    remote_path = os.path.join(APP_DIR, remote_name)
    with c.open_sftp() as s:
        s.put(local_path, remote_path)
    print(f"  → Copied {local_name} to {remote_path}", flush=True)

# ─── 1. Stop service ───
print("\n=== 1. Stop qwen-coder service ===")
sudo("systemctl stop qwen-coder 2>/dev/null; echo done")
ssh("ss -tlnp | grep 8000 || echo PORT_FREE")

# ─── 2. Copy build files to Spark ───
print("\n=== 2. Copy Dockerfile + entrypoint ===")
ssh(f"mkdir -p {APP_DIR}")
sftp_put("Dockerfile.turboquant", "Dockerfile.turboquant")
sftp_put("turboquant-entrypoint.py", "turboquant-entrypoint.py")
ssh(f"chmod +x {APP_DIR}/turboquant-entrypoint.py")

# ─── 3. Pull base image (if needed) ───
print("\n=== 3. Pull base image ===")
sudo(f"docker pull {BASE_IMAGE} 2>&1 | tail -5", t=600)

# ─── 4. Build TurboQuant-enhanced image ───
print("\n=== 4. Build TurboQuant image (~2-3 min) ===")
build_cmd = (
    f"cd {APP_DIR} && "
    f"docker build -t {NEW_TAG} "
    f"-f Dockerfile.turboquant "
    f"--network host "
    f". 2>&1 | tail -20"
)
sudo(build_cmd, t=600)

# Verify TurboQuant is installed
print("\n=== 4b. Verify TurboQuant ===")
sudo(
    f"docker run --rm --gpus all --entrypoint python3 {NEW_TAG} "
    f"-c \"from turboquant.vllm_attn_backend import enable_no_alloc; "
    f"enable_no_alloc(3,2,128); print('TurboQuant OK: 3-bit keys, 2-bit values')\"",
    t=60
)

# ─── 5. Update run script ───
print("\n=== 5. Update run_qwen.sh ===")
run_script = f"""#!/usr/bin/env bash
set -euo pipefail

exec /usr/bin/docker run --rm \\
  --name {SERVICE_NAME} \\
  --gpus all \\
  --ipc=host \\
  --network host \\
  --ulimit memlock=-1 \\
  --shm-size=16g \\
  --env-file {APP_DIR}/llm.env \\
  -v /home/{USER}/.cache/huggingface:/root/.cache/huggingface \\
  {NEW_TAG} \\
  serve \\
  --host 0.0.0.0 \\
  --port {PORT} \\
  --served-model-name qwen3-coder-next \\
  --quantization modelopt \\
  --dtype auto \\
  --gpu-memory-utilization 0.90 \\
  --max-model-len 8192 \\
  --max-num-seqs 8 \\
  --max-num-batched-tokens 8192 \\
  --moe-backend marlin \\
  --tokenizer-mode hf \\
  --chat-template-content-format string \\
  --trust-remote-code
EOF
"""
ssh(f"cat > {APP_DIR}/run_qwen.sh << 'RUNEOF'\n{run_script}\nRUNEOF")
ssh(f"chmod +x {APP_DIR}/run_qwen.sh")
print("  ✓ run_qwen.sh updated (TurboQuant image, no --kv-cache-dtype fp8)", flush=True)

# ─── 6. Reload systemd & start ───
print("\n=== 6. Start qwen-coder with TurboQuant ===")
sudo("systemctl daemon-reload")
sudo(f"systemctl start {SERVICE_NAME}")
print("  Service started, waiting for model load...", flush=True)
time.sleep(30)

# ─── 7. Monitor startup ───
print("\n=== 7. Monitor startup ===")
for attempt in range(8):
    sudo(f"systemctl status {SERVICE_NAME} --no-pager | head -8", t=15)
    ssh("ss -tlnp | grep 8000 || echo WAITING", t=10)
    # Check for TurboQuant log message
    sudo(f"docker logs {SERVICE_NAME} 2>&1 | grep -i turboquant || echo 'checking...'", t=10)
    print(f"  [{(attempt + 1) * 30}s]")
    time.sleep(30)

# ─── 8. Smoke test ───
print("\n=== 8. Smoke test ===")
ssh(
    "curl -s -X POST http://localhost:8000/v1/chat/completions "
    "-H 'Content-Type: application/json' "
    "-d '{\"model\":\"qwen3-coder-next\",\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}],\"max_tokens\":20}' 2>&1 | head -10",
    t=180
)

# ─── 9. LocalProxy restart ───
print("\n=== 9. Restart LocalProxy ===")
sudo("systemctl restart localproxy")
time.sleep(3)
ssh("curl -s http://localhost:9001/healthz 2>&1")

print("\n" + "=" * 60)
print("  TURBOQUANT DEPLOYMENT COMPLETE")
print("=" * 60)
print(f"  Image:    {NEW_TAG}")
print(f"  vLLM API: http://{HOST}:{PORT}/v1")
print(f"  KV-Cache: TurboQuant (keys=3-bit, values=2-bit)")
print(f"  statt:    fp8 (vLLM built-in)")
print("=" * 60)

c.close()
