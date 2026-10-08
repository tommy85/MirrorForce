"""Shared explicit identities for current-root search; no model, engine or replay imports."""

MEMORY_LAW = "own_live_opponent_root_empty/v1"
FIELD_KNOWLEDGE_LAW = "current-root-field-constraints;policy-anonymous-slots/v1"
NATIVE_IMPORT_LAW = "three-zone-projection-only/v1"
CAPABILITY = {
    "schema": "mirrorforce_current_root_sessions/v2",
    "memory_law": MEMORY_LAW,
    "binding": "owned-live-parent-pending; opponent-current-root-only/v1",
    "field_knowledge": FIELD_KNOWLEDGE_LAW,
    "native_import": NATIVE_IMPORT_LAW,
}
LAW = "public-current-root-uniform/v3"
PROVIDER = "own-pending-current-root/v1"
AR_LAW = "public-current-root-ar-direct/v2"
AR_PROVIDER = "own-pending-current-root-ar-diagnostic/v2"
AR_BANK_LAW = "direct-ar-same-current-public-law/v2"
AR_PUBLIC_LAW = "mirrorforce_current_public_slot_law/v2"
AR_HAND_SCOPE = "mirrorforce_same_pending_public_hand_scope/v1"
AR_HAND_ASSIGNMENT = "uniform-physical-hand-codes-given-public-anchors-and-shuffle-group/v1"
AR_HAND_UID_LAW = "conditional-uniform-old-group-UID-matching;fresh-and-public-anchors-fixed/v1"
AR_NATIVE_PROOF = "mirrorforce_direct_current_ar_native_bank/v2"
ROOT_RESPONSE_LAW = "certified-server-to-local-root-response/v1"
PARTICLE_IDENTITY = {
    "law": LAW, "provider": PROVIDER,
    "realizer": "current-root-owned-snapshots/v1",
    "memory_law": MEMORY_LAW,
    "own_deck": "public-uniform-uid-order/v1",
    "belief": "uniform-diagnostic",
    "root_response": ROOT_RESPONSE_LAW,
    "field_knowledge": FIELD_KNOWLEDGE_LAW,
}
AR_PARTICLE_IDENTITY = {
    **PARTICLE_IDENTITY,
    "law": AR_LAW,
    "provider": AR_PROVIDER,
    "belief": "direct-ar-current-public-diagnostic/v1",
    "public_law": AR_PUBLIC_LAW,
    "bank_law": AR_BANK_LAW,
    "hand_scope": AR_HAND_SCOPE,
    "hand_physical_law": AR_HAND_ASSIGNMENT,
    "hand_uid_law": AR_HAND_UID_LAW,
    "world_rpc": {"schema": "mirrorforce_client_public_world/v1",
                  "codec": "mirrorforce_public_world_codec/v1", "source": "cloned-own-pending-client/v1",
                  "binding": "expected_obs_sha256", "ownership": "client-connection/v1"},
    "search_admission": False,
    "strength_evaluation": False,
}
COUNT_LAW = "public-current-root-count-head-resampled/v1"
COUNT_PROVIDER = "own-pending-current-root-count-head/v1"
COUNT_BANK_LAW = "uniform-public-proposals-count-head-marginal-ratio-resampled/v1"
COUNT_NATIVE_PROOF = "mirrorforce_current_root_count_head_bank/v1"
COUNT_PARTICLE_IDENTITY = {
    **PARTICLE_IDENTITY,
    "law": COUNT_LAW,
    "provider": COUNT_PROVIDER,
    "belief": "same-public-forward-count-head/v1",
    "bank_law": COUNT_BANK_LAW,
    "search_admission": False,
}
ROOT_SCHEMA = "mirrorforce_current_root_search/v3"
PUBLIC_SEED_SCHEMA = "mirrorforce_current_public_root_seed/v1"
PUBLIC_SEED_RPC_SCHEMA = "mirrorforce_current_public_root_seed_rpc/v1"


def check_native_import(native):
    if getattr(native, "current_root_unpositioned_import_law", None) != NATIVE_IMPORT_LAW:
        raise ValueError("current-root search requires the guarded three-zone native projection importer")
