"""Start the whole app with one command: the three llama-servers, the gateway, then the chat.

Run it through scripts/run_app.sh (Linux) or scripts\\run_app.cmd (Windows cmd). Anything already
running on its port is reused; whatever this script started is stopped when the chat exits.
Arguments go to cli/joule.py, e.g. run_app.cmd -p "what is 17 * 23".
  LLAMA_SERVER  llama-server binary (default: the CUDA build from setup if there is an NVIDIA GPU,
                else config.yaml's CPU build)
Logs: data/logs/launch.log, data/logs/gateway.log, data/logs/llama-<tier>.log
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import load_config, repo_path  # noqa: E402

WINDOWS = os.name == "nt"
# Children write logs to files; on Windows that would otherwise use the ANSI code page and fail on τ.
ENV = {**os.environ, "PYTHONUTF8": "1"}


def healthy(port: int) -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200
    except requests.RequestException:
        return False


def has_gpu() -> bool:
    smi = shutil.which("nvidia-smi")
    return smi is not None and subprocess.run([smi, "-L"], capture_output=True).returncode == 0


def llama_server(cfg) -> Path:
    if os.environ.get("LLAMA_SERVER"):
        return Path(os.environ["LLAMA_SERVER"])
    cpu = repo_path(cfg["server"]["llama_server_bin"] + (".exe" if WINDOWS else ""))
    cuda = cpu.parent.with_name(cpu.parent.name + "-cuda") / cpu.name
    return cuda if cuda.exists() and has_gpu() else cpu


def start(script: str, args: list, log_path: Path) -> subprocess.Popen:
    # Own process group / session, so Ctrl-C in the chat's terminal does not reach the servers;
    # stop() shuts them down instead.
    group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {"start_new_session": True}
    return subprocess.Popen([sys.executable, str(repo_path(script)), *map(str, args)], env=ENV,
                            stdin=subprocess.DEVNULL, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, **group)


def stop(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    if WINDOWS:  # terminate() would leave launch.py's llama-servers running; /T kills the whole tree
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()  # launch.py stops its llama-servers on SIGTERM
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def wait_for(proc: subprocess.Popen, name: str, log_path: Path, ports: list[int]):
    for port in ports:
        while not healthy(port):
            if proc.poll() is not None:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
                sys.exit(f"{name} exited; last lines of {log_path}:\n" + "\n".join(tail))
            time.sleep(1)


def main() -> int:
    cfg = load_config()
    logs = repo_path("data/logs")
    logs.mkdir(parents=True, exist_ok=True)
    tier_ports = [m["port"] for m in cfg["models"].values()]
    gateway_port = cfg["gateway"]["port"]
    procs = []
    try:
        if all(healthy(p) for p in tier_ports):
            print("model servers: already running, reusing them")
        else:
            server = llama_server(cfg)
            if not server.exists():
                sys.exit(f"llama-server not found at {server}: run scripts/setup.sh or scripts\\setup.cmd first")
            log = logs / "launch.log"
            print(f"starting the model servers with {server} (first start loads ~4 GiB of models; log: {log})",
                  flush=True)
            procs.append(start("serving/launch.py", ["--llama-server", server], log))
            wait_for(procs[-1], "serving/launch.py", log, tier_ports)

        if healthy(gateway_port):
            print(f"gateway: already running on :{gateway_port}, reusing it")
        else:
            log = logs / "gateway.log"
            print(f"starting the gateway (log: {log})", flush=True)
            procs.append(start("gateway/app.py", [], log))
            wait_for(procs[-1], "gateway/app.py", log, [gateway_port])

        return subprocess.call([sys.executable, str(repo_path("cli/joule.py")),
                                "--url", f"http://127.0.0.1:{gateway_port}", *sys.argv[1:]], env=ENV)
    except KeyboardInterrupt:
        return 130
    finally:
        if procs:
            print("stopping the gateway and model servers", flush=True)
            for proc in reversed(procs):
                stop(proc)


if __name__ == "__main__":
    sys.exit(main())
