CREATE TABLE public.crypto_session_trades_scan (
    id SERIAL PRIMARY KEY,

    -- Timestamp of the scan run
    run_timestamp TIMESTAMPTZ NOT NULL,  -- Timezone-aware for consistency in Supabase

    -- Asset information
    primary_asset_symbol TEXT NOT NULL,       -- e.g., 'BTCUSD'
    primary_asset_price NUMERIC(18, 8) NOT NULL,
    secondary_asset_symbol TEXT NOT NULL,     -- e.g., 'ETHUSD'
    secondary_asset_price NUMERIC(18, 8) NOT NULL,

    -- Calculated metrics
    hedge_ratio NUMERIC(10, 6) NOT NULL,      -- e.g., 0.3002
    z_score NUMERIC(10, 6) NOT NULL,          -- e.g., 0.9150

    -- Strategy parameters
    entry_z_threshold NUMERIC(10, 6) NOT NULL,    -- e.g., 2.20
    exit_z_threshold NUMERIC(10, 6) NOT NULL,     -- e.g., 0.30
    stoploss_z_threshold NUMERIC(10, 6) NOT NULL, -- e.g., 3.50

    -- Trade setup
    primary_quantity NUMERIC(18, 8) NOT NULL,     -- e.g., 0.1 BTC
    secondary_quantity NUMERIC(18, 8) NOT NULL,   -- e.g., 1.277439 ETH

    -- Audit
    created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE public.crypto_session_trades_scan ENABLE ROW LEVEL SECURITY;