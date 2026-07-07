-- Silver : une ligne par tour joué, enrichie de la comparaison avec l'agent optimal.
select
    d.run_id,
    cast(d.turn as integer) as turn,
    d.direction,
    d.reason,
    d.blocked,
    d.gold,
    c.opt_dir,
    c.match,
    cast(c.opt_steps_remaining as integer) as opt_steps_remaining,
    cast(c.regret as integer)              as regret
from {{ source('bronze', 'decisions') }} d
left join {{ source('bronze', 'comparisons') }} c
    on d.run_id = c.run_id and d.turn = c.turn
