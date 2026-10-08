from mirrorforce.agent.env.registration import register

register(
  task_id="Duel-v1",
  import_path="mirrorforce.agent.env.duel",
  spec_cls="DuelEnvSpec",
  dm_cls="DuelDMEnvPool",
  gym_cls="DuelGymEnvPool",
  gymnasium_cls="DuelGymnasiumEnvPool",
)
