-- nexmark_q5.sql — NEXMark Query 5: Hot Items (sliding window bid count)
--
-- This is the primary Flink SQL job used in XTAL experiments.
-- Q5 is chosen because it is stateful (window state), exercises the RocksDB
-- state backend, and is sensitive to checkpoint interval and buffer timeout.
--
-- Submitted via: ./jobs/submit_sql_job.sh

-- ── Source: Kafka bids ────────────────────────────────────────────────────────
CREATE TABLE nexmark_bids (
    auction     BIGINT,
    bidder      BIGINT,
    price       DECIMAL(10, 2),
    channel     VARCHAR,
    url         VARCHAR,
    event_time  BIGINT,
    seq         BIGINT,
    proc_time   AS PROCTIME(),
    row_time    AS TO_TIMESTAMP_LTZ(event_time, 3),
    WATERMARK FOR row_time AS row_time - INTERVAL '2' SECOND
) WITH (
    'connector'                     = 'kafka',
    'topic'                         = 'nexmark-bids',
    'properties.bootstrap.servers'  = 'kafka:9092',
    'properties.group.id'           = 'xtal-q5-consumer',
    'scan.startup.mode'             = 'latest-offset',
    'format'                        = 'json',
    'json.ignore-parse-errors'      = 'true'
);

-- ── Source: Kafka auctions ────────────────────────────────────────────────────
CREATE TABLE nexmark_auctions (
    id           BIGINT,
    item_name    VARCHAR,
    initial_bid  DECIMAL(10, 2),
    reserve      DECIMAL(10, 2),
    date_time    BIGINT,
    expires      BIGINT,
    seller       BIGINT,
    category     INT,
    proc_time    AS PROCTIME(),
    row_time     AS TO_TIMESTAMP_LTZ(date_time, 3),
    WATERMARK FOR row_time AS row_time - INTERVAL '4' SECOND
) WITH (
    'connector'                     = 'kafka',
    'topic'                         = 'nexmark-auctions',
    'properties.bootstrap.servers'  = 'kafka:9092',
    'properties.group.id'           = 'xtal-auctions-consumer',
    'scan.startup.mode'             = 'latest-offset',
    'format'                        = 'json',
    'json.ignore-parse-errors'      = 'true'
);

-- ── Sink: Kafka output ────────────────────────────────────────────────────────
CREATE TABLE nexmark_output (
    auction       BIGINT,
    bid_count     BIGINT,
    window_start  TIMESTAMP(3),
    window_end    TIMESTAMP(3),
    emit_time     BIGINT
) WITH (
    'connector'                     = 'kafka',
    'topic'                         = 'nexmark-output',
    'properties.bootstrap.servers'  = 'kafka:9092',
    'format'                        = 'json'
);

-- ── Q5: Count bids per auction in a sliding window ───────────────────────────
-- Sliding window: size=10s, slide=2s
-- This creates significant state (each event participates in 5 windows)
-- and exercises checkpoint performance under high throughput.
INSERT INTO nexmark_output
SELECT
    auction,
    COUNT(*)                              AS bid_count,
    window_start,
    window_end,
    UNIX_TIMESTAMP() * 1000              AS emit_time
FROM TABLE(
    HOP(
        TABLE nexmark_bids,
        DESCRIPTOR(row_time),
        INTERVAL '2' SECOND,    -- slide
        INTERVAL '10' SECOND    -- size
    )
)
GROUP BY
    auction,
    window_start,
    window_end;
