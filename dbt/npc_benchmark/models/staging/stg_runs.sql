-- Silver : une ligne par run, types nettoyés.
select
    run_id,
    model,
    typology,
    cast(final_score as integer)   as final_score,
    cast(total_gold as integer)    as total_gold,
    cast(turns_used as integer)    as turns_used,
    status,
    has_initial_map,
    cast(optimal_steps as integer) as optimal_steps
from {{ source('bronze', 'runs') }}
