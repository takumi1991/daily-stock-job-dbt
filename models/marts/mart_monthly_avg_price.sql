{{ config(
    materialized='table'
) }}

-- Mart: 月次平均株価
select
  symbol,
  format_date('%Y-%m', date) as month,
  avg(close) as avg_close,
  avg(volume) as avg_volume
from {{ ref('stg_daily_prices') }}
group by symbol, month
order by symbol, month
