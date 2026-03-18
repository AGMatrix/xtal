"""
load_generator.py — Produces NEXMark-style auction/bid events to Kafka.

Usage:
    python scripts/load_generator.py --rate 5000 --topic nexmark-bids --duration 120
    python scripts/load_generator.py --rate 5000 --topic nexmark-bids --burst-at 30 --burst-rate 50000

Event schema (JSON, NEXMark Bid):
    { "auction": int, "bidder": int, "price": float,
      "channel": str, "url": str, "event_time": int (epoch ms) }
"""

import argparse
import json
import logging
import random
import string
import time
from pathlib import Path

from confluent_kafka import Producer

log = logging.getLogger(__name__)

CHANNELS = ["mobile", "desktop", "tablet", "api"]
URLS = ["https://example.com/" + "".join(random.choices(string.ascii_lowercase, k=8)) for _ in range(50)]


def make_bid(seq: int) -> dict:
    return {
        "auction": random.randint(1, 100_000),
        "bidder": random.randint(1, 1_000_000),
        "price": round(random.uniform(0.01, 10_000.0), 2),
        "channel": random.choice(CHANNELS),
        "url": random.choice(URLS),
        "event_time": int(time.time() * 1000),
        "seq": seq,
    }


def make_auction(seq: int) -> dict:
    return {
        "id": seq,
        "item_name": "item-" + str(random.randint(1, 10_000)),
        "description": "desc",
        "initial_bid": round(random.uniform(1.0, 1_000.0), 2),
        "reserve": round(random.uniform(1_000.0, 5_000.0), 2),
        "date_time": int(time.time() * 1000),
        "expires": int((time.time() + 3600) * 1000),
        "seller": random.randint(1, 100_000),
        "category": random.randint(1, 20),
    }


def load_producer_hints() -> dict:
    """Reads XTAL controller hints (e.g. linger.ms overrides)."""
    hint_file = Path("/tmp/xtal_producer_hints.json")
    if hint_file.exists():
        return json.loads(hint_file.read_text())
    return {}


def produce(args: argparse.Namespace) -> None:
    base_config = {
        "bootstrap.servers": args.bootstrap,
        "linger.ms": args.linger_ms,
        "batch.size": 65536,
        "compression.type": "lz4",
        "acks": "1",
    }

    producer = Producer(base_config)
    seq = 0
    start = time.time()
    burst_triggered = False
    current_rate = args.rate

    log.info("Starting load generator: rate=%d/s, duration=%ds, topic=%s",
             args.rate, args.duration, args.topic)

    while True:
        elapsed = time.time() - start
        if args.duration > 0 and elapsed >= args.duration:
            break

        # Burst injection
        if args.burst_at > 0 and not burst_triggered and elapsed >= args.burst_at:
            current_rate = args.burst_rate
            burst_triggered = True
            log.info("BURST TRIGGERED at t=%.1fs: rate → %d/s", elapsed, current_rate)

        # Check for XTAL controller hint overrides
        hints = load_producer_hints()
        if "linger.ms" in hints:
            new_linger = int(hints["linger.ms"])
            if new_linger != int(producer.__dict__.get("_linger_ms", args.linger_ms)):
                log.info("XTAL hint: linger.ms → %d", new_linger)
                # Recreate producer with new linger (simplest approach)
                producer.flush()
                base_config["linger.ms"] = new_linger
                producer = Producer(base_config)
                producer.__dict__["_linger_ms"] = new_linger

        # Produce events
        batch_start = time.monotonic()
        for _ in range(current_rate // 10):  # produce in 100ms windows
            seq += 1
            if seq % 20 == 0:  # 5% auctions, 95% bids
                event = make_auction(seq)
                topic = args.auction_topic
            else:
                event = make_bid(seq)
                topic = args.topic

            producer.produce(
                topic=topic,
                key=str(event.get("auction", seq)).encode(),
                value=json.dumps(event).encode(),
                on_delivery=lambda err, msg: (
                    log.warning("Delivery failed: %s", err) if err else None
                ),
            )
            producer.poll(0)

        # Rate control: sleep to fill the 100ms window
        elapsed_batch = time.monotonic() - batch_start
        sleep_time = 0.1 - elapsed_batch
        if sleep_time > 0:
            time.sleep(sleep_time)

        if seq % 10_000 == 0:
            log.info("Produced %d events in %.1fs (rate=%d/s)", seq, time.time() - start, current_rate)

    producer.flush()
    log.info("Done. Produced %d events total in %.1fs", seq, time.time() - start)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="XTAL NEXMark Load Generator")
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--topic", default="nexmark-bids")
    p.add_argument("--auction-topic", default="nexmark-auctions")
    p.add_argument("--rate", type=int, default=5_000, help="events/sec baseline")
    p.add_argument("--burst-at", type=float, default=0, help="seconds until burst (0=no burst)")
    p.add_argument("--burst-rate", type=int, default=50_000, help="events/sec during burst")
    p.add_argument("--duration", type=int, default=120, help="total seconds (0=infinite)")
    p.add_argument("--linger-ms", type=int, default=5)
    args = p.parse_args()
    produce(args)


if __name__ == "__main__":
    main()
