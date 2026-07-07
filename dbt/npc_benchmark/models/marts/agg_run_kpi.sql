-- Gold : un run = une ligne de KPI (aggrégat métier), prêt pour le reporting.
with per_run as (
    select
        run_id,
        count(*)                                as n_turns,
        sum(case when blocked then 1 else 0 end) as n_blocked,
        sum(case when gold then 1 else 0 end)    as n_gold_collected,
        sum(case when match then 1 else 0 end)   as n_match,
        sum(regret)                              as total_regret
    from {{ ref('stg_decisions') }}
    group by run_id
)

select
    r.run_id,
    r.model,
    r.typology,
    r.status,
    r.total_gold,
    r.final_score,
    r.turns_used,
    r.optimal_steps,
    p.n_turns,
    p.n_blocked,
    round(p.n_blocked * 1.0 / nullif(p.n_turns, 0), 2)               as blocked_rate,
    p.n_gold_collected,
    p.n_match,
    round(p.n_match * 1.0 / nullif(p.n_turns, 0), 2)                 as match_rate,
    p.total_regret,
    case when r.status = 'won'
         then round(r.optimal_steps * 1.0 / nullif(r.turns_used, 0), 2)
    end as efficiency,
    case when r.status = 'won'
         then r.turns_used - r.optimal_steps
    end as wasted_turns
from {{ ref('stg_runs') }} r
left join per_run p on r.run_id = p.run_id
