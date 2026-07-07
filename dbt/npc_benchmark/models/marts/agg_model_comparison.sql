-- Gold : comparaison inter-modèles LLM (axe "Modèles de LLM" du benchmark).
select
    model,
    count(*)                                                as n_runs,
    sum(case when status = 'won' then 1 else 0 end)         as n_won,
    round(avg(case when status = 'won' then 1.0 else 0 end), 2) as win_rate,
    round(avg(match_rate), 2)                               as avg_match_rate,
    round(avg(blocked_rate), 2)                              as avg_blocked_rate,
    round(avg(efficiency), 2)                                as avg_efficiency,
    round(avg(total_regret), 2)                              as avg_total_regret,
    round(avg(turns_used), 1)                                as avg_turns_used
from {{ ref('agg_run_kpi') }}
group by model
