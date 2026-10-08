// PRIVILEGED_TARGET: standalone, read-only diagnostic against the pinned core
// header layout. Never linked into a policy/follower library or used as a
// replacement for query_local_entity_map's strict validation.
#include "duel.h"
#include "card.h"
#include "field.h"
#include "interpreter.h"
#include <cstddef>

#if !defined(MF_ENTITY_PROBE_CORE_EA37B983) || _GLIBCXX_USE_CXX11_ABI != 1 || defined(_GLIBCXX_DEBUG)
#error "Build only against archived ea37b983 headers, GCC C++17 release libstdc++ ABI=1"
#endif
static_assert(sizeof(void*) == 8 && sizeof(bool) == 1, "pinned x86_64 ABI required");
static_assert(sizeof(duel) == offsetof(duel, uncopy) + sizeof(decltype(duel::uncopy)),
              "not the pinned pre-phase-audit duel layout");

extern "C" __attribute__((visibility("default")))
int diagnose_local_entity_invariant(void* raw, uint64_t* out, uint32_t capacity) {
    if (!raw || !out || capacity < 18) return -1;
    const auto* pd = static_cast<const duel*>(raw);
    for (uint32_t i = 0; i < capacity; ++i) out[i] = 0;
    out[0] = pd->cards.size();
    out[1] = pd->lua->current_state == pd->lua->lua_state;
    out[2] = pd->lua->no_action;
    out[3] = pd->lua->params.size();
    out[4] = pd->lua->call_depth;
    auto fail = [&](int reason, const card* pc, uint64_t related = 0, uint64_t size = 0) {
        out[5] = reason;
        if (pc) {
            out[6] = pc->cardid;
            out[7] = pc->owner;
            out[8] = pc->current.controler;
            out[9] = pc->current.location;
            out[10] = pc->current.sequence;
            out[11] = pc->current.position;
            out[12] = pc->overlay_target ? pc->overlay_target->cardid : 0;
            out[15] = pc->previous.controler;
            out[16] = pc->previous.location;
            out[17] = pc->previous.sequence;
        }
        out[13] = related;
        out[14] = size;
        return reason;
    };
    for (const card* pc : pd->cards) {
        if (!pc || !pc->cardid || pc->pduel != pd) return fail(1, pc);
        if (pc->owner > 1 && pc->owner != PLAYER_NONE) return fail(2, pc);
        for (const card* other : pd->cards)
            if (other != pc && other->cardid == pc->cardid) return fail(3, pc);
        if (pc->overlay_target) {
            if (!pd->cards.count(pc->overlay_target)) return fail(4, pc);
            if (pc->current.location != LOCATION_OVERLAY || pc->current.controler != PLAYER_NONE)
                return fail(5, pc);
            const auto& v = pc->overlay_target->xyz_materials;
            if (pc->current.sequence >= v.size() || v[pc->current.sequence] != pc)
                return fail(6, pc, pc->current.sequence < v.size() && v[pc->current.sequence]
                            ? v[pc->current.sequence]->cardid : 0, v.size());
        } else if (pc->current.location) {
            if (pc->current.controler > 1) return fail(7, pc);
            const auto& player = pd->game_field->player[pc->current.controler];
            const card_vector* v = nullptr;
            switch (pc->current.location) {
            case LOCATION_MZONE: v = &player.list_mzone; break;
            case LOCATION_SZONE: v = &player.list_szone; break;
            case LOCATION_DECK: v = &player.list_main; break;
            case LOCATION_HAND: v = &player.list_hand; break;
            case LOCATION_GRAVE: v = &player.list_grave; break;
            case LOCATION_REMOVED: v = &player.list_remove; break;
            case LOCATION_EXTRA: v = &player.list_extra; break;
            }
            if (!v) return fail(8, pc);
            if (pc->current.sequence >= v->size()) return fail(9, pc, 0, v->size());
            if ((*v)[pc->current.sequence] != pc)
                return fail(10, pc, (*v)[pc->current.sequence] ? (*v)[pc->current.sequence]->cardid : 0, v->size());
        } else if (pc->current.controler != PLAYER_NONE) return fail(11, pc);
    }
    return 0;
}
