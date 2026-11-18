CREATE TABLE customers (customer_id VARCHAR PRIMARY KEY);
INSERT INTO customers VALUES ('A'), ('B'), ('C');

CREATE TABLE buying_customers (customer_id VARCHAR PRIMARY KEY);
INSERT INTO buying_customers VALUES ('A'), ('B');

CREATE TABLE channels (channel VARCHAR NOT NULL, marketing_group VARCHAR NOT NULL);
INSERT INTO channels VALUES ('web', 'digital'), ('store', 'physical');

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id VARCHAR NOT NULL,
    placed_at TIMESTAMP NOT NULL,
    channel VARCHAR NOT NULL,
    gross_amount INTEGER NOT NULL,
    refunded_amount INTEGER NOT NULL
);
INSERT INTO orders VALUES
    (1, 'A', TIMESTAMP '2026-01-01 13:00:00', 'web',   20,  0),
    (2, 'A', TIMESTAMP '2026-01-01 14:00:00', 'store', 40,  5),
    (3, 'B', TIMESTAMP '2026-01-01 15:00:00', 'web',   60, 10);
