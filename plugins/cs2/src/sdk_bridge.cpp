#include "sdk_bridge.h"

#include "schema_offset.h"

#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace aegis {

static SdkHooks g_hooks;
SdkHooks& sdk_hooks() { return g_hooks; }

}  // namespace aegis

#ifdef AEGIS_WITH_METAMOD

#  include <eiface.h>
#  include <iserver.h>
#  include <schemasystem/schemasystem.h>

namespace aegis {
namespace sdk {

namespace {

constexpr int kMaxSlots = 64;

std::mutex g_pawn_mu;
std::array<void*, kMaxSlots> g_pawns{};
std::array<int, kMaxSlots> g_last_weapon{};

IVEngineServer* g_engine = nullptr;
ISource2Server* g_server = nullptr;
INetworkServerService* g_netsvc = nullptr;
float g_tick_interval = 1.0f / 64.0f;
bool g_vis_ready = false;

struct NamedWeapon {
    const char* name;
    int defindex;
};

// CS2 item definition indices (server event "weapon" names drop weapon_ prefix).
constexpr NamedWeapon kWeapons[] = {
    {"ak47", 7},         {"m4a1", 16},       {"m4a1_silencer", 60},
    {"awp", 9},          {"ssg08", 40},      {"deagle", 1},
    {"usp_silencer", 61},{"glock", 4},       {"p250", 36},
    {"tec9", 30},        {"fiveseven", 3},   {"galilar", 13},
    {"famas", 10},       {"aug", 8},         {"sg556", 39},
    {"mp9", 34},         {"mac10", 17},      {"mp7", 33},
    {"ump45", 24},       {"p90", 19},        {"bizon", 26},
    {"nova", 35},        {"xm1014", 25},     {"mag7", 27},
    {"negev", 28},       {"m249", 14},       {"scar20", 38},
    {"g3sg1", 11},       {"revolver", 64},   {"cz75a", 63},
    {"elite", 2},        {"hkp2000", 32},    {"mp5sd", 23},
    {"knife", 42},       {"hegrenade", 44},  {"flashbang", 43},
    {"smokegrenade", 45},{"molotov", 46},    {"incgrenade", 48},
    {"decoy", 47},
};

QAngle3 read_angles(void* pawn, const char* cls, const char* field) {
    QAngle3 a{};
    if (auto* p = schema::field<QAngle3>(pawn, cls, field)) return *p;
    // Alternate class names used across schema dumps.
    if (auto* p = schema::field<QAngle3>(pawn, "CCSPlayerPawnBase", field)) return *p;
    if (auto* p = schema::field<QAngle3>(pawn, "CCSPlayerPawn", field)) return *p;
    (void)a;
    return {};
}

// CS2 keeps world transform on the entity's scene node, not the entity itself:
// CBaseEntity::m_pGameSceneNode -> CGameSceneNode::m_vecAbsOrigin.
Vec3 read_scene_origin(void* pawn) {
    void** node_ptr = schema::field<void*>(pawn, "CBaseEntity", "m_pGameSceneNode");
    if (!node_ptr || !*node_ptr) return {};
    if (auto* p = schema::field<Vec3>(*node_ptr, "CGameSceneNode", "m_vecAbsOrigin")) {
        return *p;
    }
    return {};
}

QAngle3 read_scene_angles(void* pawn) {
    void** node_ptr = schema::field<void*>(pawn, "CBaseEntity", "m_pGameSceneNode");
    if (!node_ptr || !*node_ptr) return {};
    if (auto* p = schema::field<QAngle3>(*node_ptr, "CGameSceneNode", "m_angAbsRotation")) {
        return *p;
    }
    return {};
}

Vec3 read_vec(void* pawn, const char* primary_cls, const char* field) {
    if (auto* p = schema::field<Vec3>(pawn, primary_cls, field)) return *p;
    if (auto* p = schema::field<Vec3>(pawn, "CBasePlayerPawn", field)) return *p;
    if (auto* p = schema::field<Vec3>(pawn, "CBaseEntity", field)) return *p;
    if (auto* p = schema::field<Vec3>(pawn, "CCSPlayerPawn", field)) return *p;
    return {};
}

int try_int(void* pawn, std::initializer_list<std::pair<const char*, const char*>> locs,
            int fallback = 0) {
    for (auto [cls, field] : locs) {
        const int off = schema::offset_of(cls, field);
        if (off < 0) continue;
        return *reinterpret_cast<int*>(reinterpret_cast<uintptr_t>(pawn) +
                                       static_cast<uintptr_t>(off));
    }
    return fallback;
}

bool try_bool(void* pawn, std::initializer_list<std::pair<const char*, const char*>> locs) {
    for (auto [cls, field] : locs) {
        const int off = schema::offset_of(cls, field);
        if (off < 0) continue;
        return *reinterpret_cast<bool*>(reinterpret_cast<uintptr_t>(pawn) +
                                        static_cast<uintptr_t>(off));
    }
    return false;
}

float try_float(void* pawn, std::initializer_list<std::pair<const char*, const char*>> locs) {
    for (auto [cls, field] : locs) {
        const int off = schema::offset_of(cls, field);
        if (off < 0) continue;
        return *reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(pawn) +
                                         static_cast<uintptr_t>(off));
    }
    return 0.f;
}

uint32_t try_u32(void* pawn, std::initializer_list<std::pair<const char*, const char*>> locs) {
    for (auto [cls, field] : locs) {
        const int off = schema::offset_of(cls, field);
        if (off < 0) continue;
        return *reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(pawn) +
                                            static_cast<uintptr_t>(off));
    }
    return 0;
}

int read_flags_bits(void* pawn) {
    int f = 0;
    const uint32_t ent_flags = try_u32(pawn, {{"CBaseEntity", "m_fFlags"}});
    // FL_ONGROUND=1, FL_DUCKING=2 in Source.
    if (ent_flags & 1) f |= flag::ON_GROUND;
    if (ent_flags & 2) f |= flag::DUCKING;
    if (try_bool(pawn, {{"CCSPlayerPawn", "m_bIsScoped"},
                        {"CCSPlayerPawnBase", "m_bIsScoped"}})) {
        f |= flag::SCOPED;
    }
    if (try_bool(pawn, {{"CCSPlayerPawn", "m_bIsDefusing"}})) f |= flag::DEFUSING;
    const float flash = try_float(pawn, {{"CCSPlayerPawnBase", "m_flFlashDuration"},
                                         {"CCSPlayerPawn", "m_flFlashDuration"}});
    if (flash > 0.5f) f |= flag::FLASHED;
    else if (flash > 0.05f) f |= flag::BLIND_PARTIAL;
    return f;
}

}  // namespace

void cache_pawn(int slot, void* pawn) {
    if (slot < 0 || slot >= kMaxSlots) return;
    std::lock_guard<std::mutex> lock(g_pawn_mu);
    g_pawns[static_cast<size_t>(slot)] = pawn;
}

void clear_pawn(int slot) {
    if (slot < 0 || slot >= kMaxSlots) return;
    std::lock_guard<std::mutex> lock(g_pawn_mu);
    g_pawns[static_cast<size_t>(slot)] = nullptr;
    g_last_weapon[static_cast<size_t>(slot)] = 0;
}

void* pawn_for_slot(int slot) {
    if (slot < 0 || slot >= kMaxSlots) return nullptr;
    std::lock_guard<std::mutex> lock(g_pawn_mu);
    return g_pawns[static_cast<size_t>(slot)];
}

void set_last_weapon(int slot, int defindex) {
    if (slot < 0 || slot >= kMaxSlots) return;
    g_last_weapon[static_cast<size_t>(slot)] = defindex;
}

int last_weapon(int slot) {
    if (slot < 0 || slot >= kMaxSlots) return 0;
    return g_last_weapon[static_cast<size_t>(slot)];
}

int weapon_defindex_from_name(const char* name) {
    if (!name || !*name) return 0;
    std::string n = name;
    if (n.rfind("weapon_", 0) == 0) n = n.substr(7);
    for (const auto& w : kWeapons) {
        if (n == w.name) return w.defindex;
    }
    return 0;
}

PlayerState snapshot_player(int slot, void* pawn) {
    PlayerState s;
    s.slot = slot;
    if (!pawn) return s;

    Vec3 o = read_scene_origin(pawn);
    if (o.x == 0 && o.y == 0 && o.z == 0) {
        o = read_vec(pawn, "CBasePlayerPawn", "m_vOldOrigin");
    }
    const Vec3 v = read_vec(pawn, "CBaseEntity", "m_vecAbsVelocity");
    QAngle3 eye = read_angles(pawn, "CCSPlayerPawn", "m_angEyeAngles");
    if (eye.x == 0 && eye.y == 0) {
        eye = read_angles(pawn, "CBasePlayerPawn", "v_angle");
    }
    if (eye.x == 0 && eye.y == 0) {
        eye = read_scene_angles(pawn);
    }

    s.x = o.x; s.y = o.y; s.z = o.z;
    s.vx = v.x; s.vy = v.y; s.vz = v.z;
    s.pitch = eye.pitch();
    s.yaw = eye.yaw();
    s.hp = try_int(pawn, {{"CBaseEntity", "m_iHealth"}});
    s.armor = try_int(pawn, {{"CCSPlayerPawn", "m_ArmorValue"},
                             {"CCSPlayerPawnBase", "m_ArmorValue"}});
    s.weapon_defindex = last_weapon(slot);
    s.ammo = 0;
    s.flags = read_flags_bits(pawn);
    s.shots_fired = try_int(pawn, {{"CCSPlayerPawn", "m_iShotsFired"},
                                   {"CCSPlayerPawnBase", "m_iShotsFired"}});
    s.aim_target = -1;
    s.aim_target_vis = 0;
    s.move_fwd = 0;
    s.move_side = 0;
    s.buttons = 0;
    return s;
}

FireSnapshot snapshot_fire(void* pawn) {
    FireSnapshot fs;
    if (!pawn) return fs;
    QAngle3 eye = read_angles(pawn, "CCSPlayerPawn", "m_angEyeAngles");
    if (eye.x == 0 && eye.y == 0) {
        eye = read_angles(pawn, "CCSPlayerPawnBase", "m_angEyeAngles");
    }
    QAngle3 punch = read_angles(pawn, "CCSPlayerPawn", "m_aimPunchAngle");
    if (punch.x == 0 && punch.y == 0) {
        punch = read_angles(pawn, "CCSPlayerPawnBase", "m_aimPunchAngle");
    }
    fs.pitch = eye.pitch();
    fs.yaw = eye.yaw();
    fs.punch_pitch = punch.pitch();
    fs.punch_yaw = punch.yaw();
    fs.spread = 0.f;
    return fs;
}

float dist_between_slots(int a, int b) {
    void* pa = pawn_for_slot(a);
    void* pb = pawn_for_slot(b);
    if (!pa || !pb) return -1.f;
    const PlayerState sa = snapshot_player(a, pa);
    const PlayerState sb = snapshot_player(b, pb);
    const float dx = sa.x - sb.x, dy = sa.y - sb.y, dz = sa.z - sb.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::vector<PlayerState> collect_players_impl() {
    std::vector<PlayerState> out;
    std::array<void*, kMaxSlots> snap{};
    {
        std::lock_guard<std::mutex> lock(g_pawn_mu);
        snap = g_pawns;
    }
    for (int i = 0; i < kMaxSlots; ++i) {
        if (!snap[static_cast<size_t>(i)]) continue;
        out.push_back(snapshot_player(i, snap[static_cast<size_t>(i)]));
    }
    return out;
}

bool trace_visible_impl(const PlayerState&, const PlayerState&) {
    return false;  // TraceShape not in public SDK — leave vis disabled.
}

void set_engine_context(void* engine, void* server, void* server_config,
                        void* network_service, float tick_interval) {
    (void)server_config;  // deliberately unused; see tickrate hook
    g_engine = static_cast<IVEngineServer*>(engine);
    g_server = static_cast<ISource2Server*>(server);
    g_netsvc = static_cast<INetworkServerService*>(network_service);
    if (tick_interval > 0.f) g_tick_interval = tick_interval;
}

void install_sdk_hooks() {
    auto& h = sdk_hooks();
    h.collect_players = collect_players_impl;
    h.trace_visible = trace_visible_impl;
    h.vis_ready = [] { return g_vis_ready; };
    h.current_map = []() -> std::string {
        if (g_server) {
            if (const char* m = g_server->GetActiveWorldName()) return m;
        }
        if (g_netsvc && g_netsvc->GetIGameServer()) {
            if (CGlobalVars* g = g_netsvc->GetIGameServer()->GetGlobals()) {
                if (const char* m = STRING(g->mapname)) return m;
            }
        }
        return {};
    };
    // curtime == tickcount * tick_interval, so the ratio recovers the interval.
    // ISource2ServerConfig::GetTickInterval() would be direct, but MetaMod patches
    // that interface's vtable during load and fetching it corrupts server startup.
    h.tickrate = [] {
        if (g_netsvc && g_netsvc->GetIGameServer()) {
            if (CGlobalVars* g = g_netsvc->GetIGameServer()->GetGlobals()) {
                if (g->tickcount > 8 && g->curtime > 0.f) {
                    const float iv = g->curtime / static_cast<float>(g->tickcount);
                    if (iv > 0.f) return 1.0f / iv;
                }
            }
        }
        return g_tick_interval > 0.f ? 1.0f / g_tick_interval : 64.0f;
    };
    h.steamid64_for_slot = [](int slot) -> uint64_t {
        if (!g_engine || slot < 0) return 0;
        return g_engine->GetClientXUID(CPlayerSlot(slot));
    };
    g_vis_ready = false;
}

void mark_vis_ready(bool ok) { g_vis_ready = ok; }

}  // namespace sdk
}  // namespace aegis

#endif  // AEGIS_WITH_METAMOD
