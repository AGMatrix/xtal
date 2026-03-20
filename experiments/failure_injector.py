"""
failure_injector.py — Controlled failure injection for XTAL experiments.

Failure modes:
  1. node-failure    : Stop a Flink TaskManager container
  2. network-delay   : Add 200ms delay on the kafka network interface (tc netem)
  3. cpu-stress      : Run stress-ng on a TaskManager container
  4. checkpoint-hang : Pause a TaskManager to simulate checkpoint timeout

Usage:
    python -m experiments.failure_injector node-failure --duration 30
    python -m experiments.failure_injector network-delay --delay-ms 200 --duration 60
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time

log = logging.getLogger(__name__)

TM_CONTAINERS = ["xtal-taskmanager-1", "xtal-taskmanager-2", "xtal-taskmanager-3"]


def node_failure(target: str = "xtal-taskmanager-3", duration_s: int = 30) -> None:
    """Stops a TaskManager and restarts it after duration_s seconds."""
    log.warning("FAILURE: stopping %s for %ds", target, duration_s)
    subprocess.run(["docker", "stop", target], check=True)
    time.sleep(duration_s)
    log.info("RECOVERY: restarting %s", target)
    subprocess.run(["docker", "start", target], check=True)
    log.info("Recovery complete")


def network_delay(
    container: str = "xtal-kafka",
    delay_ms: int = 200,
    jitter_ms: int = 50,
    duration_s: int = 60,
) -> None:
    """
    Adds artificial network delay inside a container using tc netem.
    Simulates a noisy-neighbor or degraded network link.
    """
    iface = "eth0"
    add_cmd = [
        "docker", "exec", container,
        "tc", "qdisc", "add", "dev", iface, "root", "netem",
        "delay", f"{delay_ms}ms", f"{jitter_ms}ms", "distribution", "normal",
    ]
    del_cmd = [
        "docker", "exec", container,
        "tc", "qdisc", "del", "dev", iface, "root",
    ]

    log.warning("FAILURE: adding %dms±%dms network delay on %s/%s for %ds",
                delay_ms, jitter_ms, container, iface, duration_s)
    try:
        subprocess.run(add_cmd, check=True)
        time.sleep(duration_s)
    finally:
        log.info("RECOVERY: removing network delay from %s/%s", container, iface)
        subprocess.run(del_cmd, check=False)


def cpu_stress(
    target: str = "xtal-taskmanager-1",
    workers: int = 2,
    duration_s: int = 30,
) -> None:
    """
    Runs CPU stress on a TaskManager to simulate noisy-neighbor CPU contention.
    Requires stress-ng to be available in the container (install via apt in Dockerfile if needed).
    """
    log.warning("FAILURE: CPU stress on %s (%d workers) for %ds", target, workers, duration_s)
    cmd = [
        "docker", "exec", "-d", target,
        "stress-ng", "--cpu", str(workers), "--timeout", f"{duration_s}s",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        log.warning("stress-ng not available in container %s — skipping CPU stress", target)
    time.sleep(duration_s)
    log.info("CPU stress complete on %s", target)


def checkpoint_hang(
    target: str = "xtal-taskmanager-2",
    pause_s: int = 15,
) -> None:
    """
    Sends SIGSTOP to a TaskManager to simulate a checkpoint hang / GC pause.
    This causes checkpoint alignment to block and p99 to spike.
    """
    log.warning("FAILURE: pausing %s for %ds (checkpoint hang simulation)", target, pause_s)
    subprocess.run(["docker", "pause", target], check=True)
    time.sleep(pause_s)
    log.info("RECOVERY: unpausing %s", target)
    subprocess.run(["docker", "unpause", target], check=True)
    log.info("Checkpoint hang simulation complete")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="XTAL Failure Injector")
    sub = p.add_subparsers(dest="mode", required=True)

    nf = sub.add_parser("node-failure")
    nf.add_argument("--target", default="xtal-taskmanager-3")
    nf.add_argument("--duration", type=int, default=30)

    nd = sub.add_parser("network-delay")
    nd.add_argument("--container", default="xtal-kafka")
    nd.add_argument("--delay-ms", type=int, default=200)
    nd.add_argument("--jitter-ms", type=int, default=50)
    nd.add_argument("--duration", type=int, default=60)

    cs = sub.add_parser("cpu-stress")
    cs.add_argument("--target", default="xtal-taskmanager-1")
    cs.add_argument("--workers", type=int, default=2)
    cs.add_argument("--duration", type=int, default=30)

    ch = sub.add_parser("checkpoint-hang")
    ch.add_argument("--target", default="xtal-taskmanager-2")
    ch.add_argument("--pause", type=int, default=15)

    args = p.parse_args()

    if args.mode == "node-failure":
        node_failure(args.target, args.duration)
    elif args.mode == "network-delay":
        network_delay(args.container, args.delay_ms, args.jitter_ms, args.duration)
    elif args.mode == "cpu-stress":
        cpu_stress(args.target, args.workers, args.duration)
    elif args.mode == "checkpoint-hang":
        checkpoint_hang(args.target, args.pause)


if __name__ == "__main__":
    main()
