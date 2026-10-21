-- discover_models: list the models available to simulate against, as compact JSON.
-- {{catalog}}/{{schema}} are substituted by 00_setup.py before execution.
CREATE OR REPLACE FUNCTION {{catalog}}.{{schema}}.discover_models()
RETURNS STRING
LANGUAGE SQL
READS SQL DATA
COMMENT 'List the models available to simulate against (model_name, version, description, num_scenarios) from model_specs, as a compact JSON array. Call this to resolve which model a run uses before reading its results.'
RETURN (
  SELECT to_json(named_struct(
    'record_type', 'model_list',
    'tool_name', 'discover_models',
    'models', sort_array(collect_list(named_struct(
      'model_name', CAST(model_name AS STRING),
      'version', CAST(version AS STRING),
      'description', CAST(description AS STRING),
      'num_scenarios', CAST(num_scenarios AS INT))))))
  FROM {{catalog}}.{{schema}}.model_specs
)
