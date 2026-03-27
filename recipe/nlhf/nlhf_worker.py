from recipe.refplay.refplay_worker import RefPlayActorRolloutRefWorker


class NLHFActorRolloutRefWorker(RefPlayActorRolloutRefWorker):
    """Dual-policy NLHF worker.

    This intentionally reuses the hardened RefPlay worker implementation:
    trainable actor+rollout on one side and rollout/ref-only for the anchor.
    """

    pass

