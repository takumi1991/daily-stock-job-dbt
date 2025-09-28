{{ config(
    materialized='view'
) }}

-- ステージングモデル: 元テーブルを正規化
select
  upper(symbol)        as symbol,
  cast(date as date)   as date,
  cast(open as float64)  as open,
  cast(high as float64)  as high,
  cast(low as float64)   as low,
  cast(close as float64) as close,
  cast(volume as int64)  as volume
from `{{ var("project_id", "stock-data-portfolio") }}.{{ var("dataset", "us_stock") }}.daily_prices`
