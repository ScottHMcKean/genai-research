-- read_runs: simulation run metadata/status as one compact JSON object.
-- {{catalog}}/{{schema}} are substituted by 00_setup.py before execution.
CREATE OR REPLACE FUNCTION {{catalog}}.{{schema}}.read_runs(
    run_ids ARRAY<STRING> DEFAULT NULL
        COMMENT 'Exact run_id values to return. NULL or empty returns all runs (most recent first).',
    p_limit INT DEFAULT 100
        COMMENT 'Maximum runs folded into the data map (capped at 500).'
)
RETURNS STRING
LANGUAGE SQL
READS SQL DATA
COMMENT 'Read simulation run metadata/status from simulation_runs as a single compact JSON object: a metadata block plus a data map keyed by run_id (model, scenario_set, status, submitted_at, completed_at, note). Runs are ordered most-recent first before the limit is applied.'
RETURN
WITH lim AS (
  SELECT greatest(1, least(coalesce(p_limit, 100), 500)) AS n
),
filtered AS (
  SELECT
    CAST(run_id AS STRING)        AS run_id,
    CAST(model AS STRING)         AS model,
    CAST(scenario_set AS STRING)  AS scenario_set,
    CAST(status AS STRING)        AS status,
    CAST(submitted_at AS STRING)  AS submitted_at,
    CAST(completed_at AS STRING)  AS completed_at,
    CAST(note AS STRING)          AS note
  FROM {{catalog}}.{{schema}}.simulation_runs
  WHERE run_id IS NOT NULL
    AND (run_ids IS NULL OR size(run_ids) = 0
         OR array_contains(run_ids, CAST(run_id AS STRING)))
),
ordered AS (
  SELECT f.*, row_number() OVER (ORDER BY submitted_at DESC NULLS LAST, run_id ASC) AS rn
  FROM filtered f
),
limited AS (
  SELECT o.* FROM ordered o CROSS JOIN lim WHERE o.rn <= lim.n
),
datamap AS (
  SELECT map_from_entries(collect_list(struct(
      run_id AS k,
      named_struct(
        'model', model, 'scenario_set', scenario_set, 'status', status,
        'submitted_at', submitted_at, 'completed_at', completed_at, 'note', note
      ) AS v))) AS m
  FROM limited
)
SELECT CASE
  WHEN (SELECT count(*) FROM filtered) = 0 THEN
    to_json(named_struct(
      'record_type', 'metadata', 'tool_name', 'read_runs',
      'requested_run_ids', run_ids, 'row_count', 0,
      'message', 'No runs matched the requested run_ids. Do not invent run_ids, statuses, or timestamps.'))
  ELSE
    to_json(named_struct(
      'record_type', 'read_runs_result', 'tool_name', 'read_runs',
      'requested_run_ids', run_ids,
      'row_count', (SELECT count(*) FROM limited),
      'total_matching', (SELECT count(*) FROM filtered),
      'data_layout', 'data[run_id] = {model, scenario_set, status, submitted_at, completed_at, note}',
      'data', (SELECT m FROM datamap),
      'message', 'Run metadata keyed by run_id. Pair these run_ids with read_sim to read metric responses.'))
  END
