-- RecoverIQ database schema
-- Run with: psql -d recoveriq -f schema.sql

CREATE TABLE IF NOT EXISTS merchants (
    merchant_id         VARCHAR(20) PRIMARY KEY,
    merchant_name       VARCHAR(100) NOT NULL,
    category            VARCHAR(50) NOT NULL,
    avg_transaction     NUMERIC(12,2) NOT NULL,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id             VARCHAR(20) PRIMARY KEY,
    account_age_days        INTEGER NOT NULL,
    previous_successes      INTEGER NOT NULL DEFAULT 0,
    previous_failures       INTEGER NOT NULL DEFAULT 0,
    avg_payment_amount      NUMERIC(12,2) NOT NULL,
    consent_status          BOOLEAN NOT NULL DEFAULT TRUE,
    preferred_payment_method VARCHAR(20),
    created_at              TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS payments (
    payment_id          VARCHAR(20) PRIMARY KEY,
    merchant_id         VARCHAR(20) REFERENCES merchants(merchant_id),
    customer_id         VARCHAR(20) REFERENCES customers(customer_id),
    amount              NUMERIC(12,2) NOT NULL,
    payment_method      VARCHAR(20) NOT NULL,       -- UPI_AUTOPAY, CARD, UPI, NETBANKING
    status              VARCHAR(20) NOT NULL,       -- FAILED, SUCCESS, RECOVERED, UNRECOVERABLE
    failure_reason      VARCHAR(30),                -- BANK_TIMEOUT, EXPIRED_CARD, INSUFFICIENT_FUNDS,
                                                      -- NETWORK_ERROR, MANDATE_AUTH_FAILED, CHECKOUT_ABANDONED, UNKNOWN
    retry_count         INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMP NOT NULL,
    last_attempt_at     TIMESTAMP,
    is_evaluation_set    BOOLEAN NOT NULL DEFAULT FALSE   -- held-out set flag
);

CREATE TABLE IF NOT EXISTS recovery_actions (
    action_id           SERIAL PRIMARY KEY,
    payment_id          VARCHAR(20) REFERENCES payments(payment_id),
    action_type         VARCHAR(30) NOT NULL,   -- RETRY_NOW, RETRY_LATER, PAYMENT_LINK, REMINDER, ESCALATE, NO_ACTION
    predicted_probability NUMERIC(5,4),
    expected_value      NUMERIC(12,2),
    chosen_by           VARCHAR(20),            -- BASELINE, AI_AGENT
    policy_decision     VARCHAR(20),            -- ALLOW, REVIEW, BLOCK
    result              VARCHAR(20),            -- SUCCESS, FAILED, PENDING
    agent_reason        TEXT,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS events (
    event_id        SERIAL PRIMARY KEY,
    payment_id      VARCHAR(20) REFERENCES payments(payment_id),
    event_type      VARCHAR(40) NOT NULL,   -- PAYMENT_FAILED, AI_ANALYSIS_STARTED, AI_RECOMMENDATION,
                                              -- POLICY_APPROVED, POLICY_BLOCKED, AGENT_REPLANNED,
                                              -- ACTION_EXECUTED, PAYMENT_RECOVERED, RECOVERY_FAILED
    payload         JSONB,
    source          VARCHAR(20),             -- SYSTEM, AGENT, POLICY, SIMULATOR
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id                SERIAL PRIMARY KEY,
    max_retries              INTEGER NOT NULL DEFAULT 3,
    min_retry_interval_hours NUMERIC(6,2) NOT NULL DEFAULT 24,
    max_action_amount        NUMERIC(12,2) NOT NULL DEFAULT 50000,
    optimize_for             VARCHAR(20) NOT NULL DEFAULT 'BALANCED', -- MAX_RECOVERY, MIN_FRICTION, BALANCED
    updated_at                TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_payments_status ON payments(status);
CREATE INDEX IF NOT EXISTS idx_payments_customer ON payments(customer_id);
CREATE INDEX IF NOT EXISTS idx_events_payment ON events(payment_id);