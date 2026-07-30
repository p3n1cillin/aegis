#include "sdk_bridge.h"

namespace aegis {

static SdkHooks g_hooks;

SdkHooks& sdk_hooks() { return g_hooks; }

}  // namespace aegis

#ifdef AEGIS_WITH_METAMOD

// ============================================================================
// SDK-backed implementations — compiled only with hl2sdk-cs2 + metamod.
// Exact schema field names and TraceShape signatures vary by CS2 build;
// the // TODO(schema) markers are the fill-in points after you sync your
// SDK to the server branch you ship.
// ============================================================================

#  include <cmath>
#  include <cstring>

// Forward-declared engine types (real headers come from HL2SDKCS2).
// Keeping them opaque here lets this translation unit document the contract
// even when include paths differ across MMS versions.
struct CBaseEntity;
struct CBasePlayerController;
struct CCSPlayerPawn;
struct CCSPlayerController;

namespace aegis {
namespace sdk {

// ---- schema field accessors (fill offsets / schema names) ---------------
//
// Recommended approach on CS2: use the schema system
//   schema::GetOffset("CCSPlayerPawn", "m_angEyeAngles")
// rather than hard-coded offsets, which break every Valve patch.

struct EyeAngles {
    float pitch = 0, yaw = 0, roll = 0;
};
struct Vec3 {
    float x = 0, y = 0, z = 0;
};

// TODO(schema): read m_angEyeAngles from CCSPlayerPawn
static EyeAngles read_eye_angles(CCSPlayerPawn* /*pawn*/) {
    return {};
}

// TODO(schema): read m_aimPunchAngle (accumulated) from CCSPlayerPawn
static EyeAngles read_aim_punch(CCSPlayerPawn* /*pawn*/) {
    return {};
}

// TODO(schema): origin / velocity / health / armor / weapon defindex / buttons
static Vec3 read_origin(CCSPlayerPawn* /*pawn*/) { return {}; }
static Vec3 read_velocity(CCSPlayerPawn* /*pawn*/) { return {}; }
static int read_health(CCSPlayerPawn* /*pawn*/) { return 0; }
static int read_armor(CCSPlayerPawn* /*pawn*/) { return 0; }
static int read_weapon_defindex(CCSPlayerPawn* /*pawn*/) { return 0; }
static int read_ammo(CCSPlayerPawn* /*pawn*/) { return 0; }
static int read_buttons(CCSPlayerPawn* /*pawn*/) { return 0; }
static int read_shots_fired(CCSPlayerPawn* /*pawn*/) { return 0; }
static float read_move_fwd(CCSPlayerPawn* /*pawn*/) { return 0; }
static float read_move_side(CCSPlayerPawn* /*pawn*/) { return 0; }

// TODO(schema): assemble PLUGIN_PROTOCOL flag bits from pawn state
static int read_flags(CCSPlayerPawn* /*pawn*/) { return 0; }

// TODO(engine): iterate controllers / pawns for the current match
static std::vector<CCSPlayerPawn*> all_pawns() { return {}; }
static int slot_of(CCSPlayerPawn* /*pawn*/) { return -1; }
static uint64_t steamid64_of(CCSPlayerController* /*ctrl*/) { return 0; }
static CCSPlayerController* controller_of(CCSPlayerPawn* /*pawn*/) { return nullptr; }

// TODO(engine): TraceShape from eye to subject. Must NOT invent visibility.
static bool trace_los(const Vec3& /*from*/, const Vec3& /*to*/) {
    return false;
}

static bool g_vis_ready = false;

std::vector<PlayerState> collect_players_impl() {
    std::vector<PlayerState> out;
    for (CCSPlayerPawn* pawn : all_pawns()) {
        if (!pawn) continue;
        PlayerState s;
        s.slot = slot_of(pawn);
        if (s.slot < 0) continue;
        const Vec3 o = read_origin(pawn);
        const Vec3 v = read_velocity(pawn);
        const EyeAngles eye = read_eye_angles(pawn);
        s.x = o.x; s.y = o.y; s.z = o.z;
        s.vx = v.x; s.vy = v.y; s.vz = v.z;
        s.pitch = eye.pitch;
        s.yaw = eye.yaw;
        s.hp = read_health(pawn);
        s.armor = read_armor(pawn);
        s.weapon_defindex = read_weapon_defindex(pawn);
        s.ammo = read_ammo(pawn);
        s.flags = read_flags(pawn);
        s.shots_fired = read_shots_fired(pawn);
        s.aim_target = -1;       // TODO(engine): crosshair raycast slot
        s.aim_target_vis = 0;
        s.move_fwd = read_move_fwd(pawn);
        s.move_side = read_move_side(pawn);
        s.buttons = read_buttons(pawn);
        out.push_back(s);
    }
    return out;
}

bool trace_visible_impl(const PlayerState& observer, const PlayerState& subject) {
    // Eye height ≈ 64 Hammer units when standing — refine from pawn view offset.
    Vec3 from{observer.x, observer.y, observer.z + 64.0f};
    Vec3 to{subject.x, subject.y, subject.z + 48.0f};
    return trace_los(from, to);
}

void install_sdk_hooks() {
    auto& h = sdk_hooks();
    h.collect_players = collect_players_impl;
    h.trace_visible = trace_visible_impl;
    h.vis_ready = [] { return g_vis_ready; };
    h.current_map = [] { return std::string{}; /* TODO(engine) */ };
    h.tickrate = [] { return 64.0f; /* TODO(engine): gpGlobals / CVAR */ };
    h.steamid64_for_slot = [](int /*slot*/) -> uint64_t {
        // TODO(engine): look up controller by slot
        return 0;
    };
    // Flip only after a successful TraceShape self-test on map start.
    g_vis_ready = false;
}

void mark_vis_ready(bool ok) { g_vis_ready = ok; }

// Snapshot punch + eye at the moment of fire. Call from weapon_fire handler
// BEFORE any post-shot angle mutation.
struct FireSnapshot {
    float pitch = 0, yaw = 0;
    float punch_pitch = 0, punch_yaw = 0;
    int weapon_defindex = 0;
    float spread = 0;
};

FireSnapshot snapshot_fire(CCSPlayerPawn* pawn) {
    FireSnapshot fs;
    if (!pawn) return fs;
    const EyeAngles eye = read_eye_angles(pawn);
    const EyeAngles punch = read_aim_punch(pawn);
    fs.pitch = eye.pitch;
    fs.yaw = eye.yaw;
    fs.punch_pitch = punch.pitch;
    fs.punch_yaw = punch.yaw;
    fs.weapon_defindex = read_weapon_defindex(pawn);
    // TODO(schema): weapon inaccuracy / spread at fire
    fs.spread = 0.0f;
    return fs;
}

}  // namespace sdk
}  // namespace aegis

#endif  // AEGIS_WITH_METAMOD
