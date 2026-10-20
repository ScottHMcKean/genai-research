-- read_sim: metric responses to scenarios over steps, as one compact JSON object.
-- Discovery mode (metrics NULL/empty) returns metadata only: the scenarios, metrics, and
-- step range available for the requested run(s). Retrieval mode (metrics given) returns a
-- nested data map: data[scenario][metric] = array of values aligned to the shared steps
-- array. {{catalog}}/{{schema}} are substituted by 00_setup.py before execution.
CREATE OR REPLACE FUNCTION {{catalog}}.{{schema}}.read_sim(
    run_ids   ARRAY<STRING>
        COMMENT 'Exact run_id values to read, e.g. ["run_0001"]. Required.',
    scenarios ARRAY<STRING> DEFAULT NULL
        COMMENT 'Exact scenario names to read. NULL or empty reads all scenarios in scope.',
    metrics   ARRAY<STRING> DEFAULT NULL
        COMMENT 'Exact metric names to read. NULL or empty = discovery mode (metadata only).',
    step_min  INT DEFAULT NULL COMMENT 'First step inclusive. NULL = no lower bound.',
    step_max  INT DEFAULT NULL COMMENT 'Last step inclusive. NULL = no upper bound.',
    p_limit   INT DEFAULT 2000 COMMENT 'Max data points folded into the map (capped at 20000).'
)
RETURNS STRING
LANGUAGE SQL
READS SQL DATA
COMMENT 'Read metric responses from simulation_output as a single compact JSON object. Metadata block plus, in retrieval mode, a nested data map data[scenario][metric] = values aligned to the shared steps array. In discovery mode (metrics NULL/empty) returns metadata only so you can choose exact scenarios/metrics, then call again with metrics populated.'
RETURN
WITH filtered AS (
  SELECT
    CAST(scenario AS STRING) AS scenario,
    CAST(metric AS STRING)   AS metric,
    CAST(step AS BIGINT)     AS step,
    CAST(value AS DOUBLE)    AS value
  FROM {{catalog}}.{{schema}}.simulation_output
  WHERE run_id IS NOT NULL AND scenario IS NOT NULL AND metric IS NOT NULL
    AND step IS NOT NULL AND value IS NOT NULL
    AND array_contains(run_ids, CAST(run_id AS STRING))
    AND (scenarios IS NULL OR size(scenarios) = 0 OR array_contains(scenarios, CAST(scenario AS STRING)))
    AND (metrics   IS NULL OR size(metrics)   = 0 OR array_contains(metrics,   CAST(metric AS STRING)))
    AND (step_min IS NULL OR step >= step_min)
    AND (step_max IS NULL OR step <= step_max)
),
stats AS (
  SELECT
    count(*) AS total_matching,
    sort_array(collect_set(scenario)) AS scenarios_found,
    sort_array(collect_set(metric))   AS metrics_found,
    min(step) AS step_min_found,
    max(step) AS step_max_found,
    greatest(1, least(coalesce(p_limit, 2000), 20000)) AS row_limit
  FROM filtered
),
-- In retrieval mode only, cap rows then build the aligned nested map.
capped AS (
  SELECT f.scenario, f.metric, f.step, f.value
  FROM (
    SELECT f.*, row_number() OVER (ORDER BY scenario, metric, step) AS rn
    FROM filtered f
    WHERE NOT (metrics IS NULL OR size(metrics) = 0)
  ) f CROSS JOIN stats s
  WHERE f.rn <= s.row_limit
),
agg AS (  -- one value per (scenario, metric, step)
  SELECT scenario, metric, step, max(value) AS value
  FROM capped GROUP BY scenario, metric, step
),
steps AS (
  SELECT sort_array(array_distinct(collect_list(step))) AS steps_arr FROM agg
),
pv AS (  -- step -> value map per (scenario, metric)
  SELECT scenario, metric,
         map_from_entries(collect_list(struct(step AS k, value AS v))) AS pvmap
  FROM agg GROUP BY scenario, metric
),
aligned AS (  -- align each metric's values to the shared steps array
  SELECT pv.scenario, pv.metric,
         transform(s.steps_arr, x -> element_at(pv.pvmap, x)) AS vals
  FROM pv CROSS JOIN steps s
),
by_metric AS (
  SELECT scenario, map_from_entries(collect_list(struct(metric AS k, vals AS v))) AS mmap
  FROM aligned GROUP BY scenario
),
datamap AS (
  SELECT map_from_entries(collect_list(struct(scenario AS k, mmap AS v))) AS m FROM by_metric
)
SELECT CASE
  WHEN s.total_matching = 0 THEN
    to_json(named_struct(
      'record_type', 'metadata', 'tool_name', 'read_sim',
      'requested_run_ids', run_ids, 'requested_scenarios', scenarios, 'requested_metrics', metrics,
      'row_count', 0,
      'message', 'No rows matched the exact filters. Do not invent scenarios, metrics, steps, or values. Check run_ids, scenarios, and metrics.'))
  WHEN metrics IS NULL OR size(metrics) = 0 THEN
    to_json(named_struct(
      'record_type', 'metadata', 'tool_name', 'read_sim', 'discovery_mode', true,
      'requested_run_ids', run_ids,
      'scenarios', s.scenarios_found, 'metrics', s.metrics_found,
      'step_min', s.step_min_found, 'step_max', s.step_max_found,
      'total_matching', s.total_matching,
      'message', 'Discovery mode: metadata only. Choose exact scenarios and metrics, then call read_sim again with metrics populated for numeric values.'))
  ELSE
    to_json(named_struct(
      'record_type', 'read_sim_result', 'tool_name', 'read_sim', 'discovery_mode', false,
      'requested_run_ids', run_ids,
      'scenarios', s.scenarios_found, 'metrics', s.metrics_found,
      'step_min', s.step_min_found, 'step_max', s.step_max_found,
      'total_matching', s.total_matching, 'row_limit', s.row_limit,
      'steps', (SELECT steps_arr FROM steps),
      'data_layout', 'data[scenario][metric] = array of values aligned to the steps array.',
      'data', (SELECT m FROM datamap),
      'message', 'Retrieval mode: data is nested scenario -> metric -> values aligned to the steps array.'))
  END
FROM stats s
