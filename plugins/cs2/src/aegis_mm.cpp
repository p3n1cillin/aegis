#include "aegis_mm.h"

#include "frames.h"
#include "salt.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace aegis {

static std::string make_session_id() {
    using clock = std::chrono::system_clock;
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        clock::now().time_since_epoch())
                        .count();
    char buf[64];
    std::snprintf(buf, sizeof(buf), "cs2-%lld", static_cast<long long>(ms));
    return std::string(buf);
}

void PluginRuntime::load(const Config& cfg) {
    cfg_ = cfg;
    net_.configure(cfg_.ingest_host, cfg_.ingest_port);
    net_.start();
    vis_.configure(cfg_.vis_hz, cfg_.vis_pair_min_interval_s);
    // Only enable vis emission when the operator opts in AND traces work.
    vis_.set_enabled(false);
    active_ = false;
}

void PluginRuntime::unload() {
    on_map_end();
    net_.stop();
}

void PluginRuntime::on_map_start(const std::string& map, float tickrate) {
    session_id_ = make_session_id();
    roster_.clear();
    seq_ = 1;
    active_ = true;

    // Prefer honest no_vis until TraceShape is wired.
    const bool no_vis = !cfg_.emit_vis || !vis_.enabled();
    auto hdr = encode_hdr(session_id_, map, tickrate, cfg_.server_name,
                          cfg_.plugin_version, no_vis, next_seq());
    net_.push(frame_packet(hdr));
}

void PluginRuntime::on_map_end() {
    active_ = false;
    roster_.clear();
}

void PluginRuntime::on_player_join(const PlayerInfoFrame& info) {
    if (!active_) return;
    // Replace existing slot entry if present.
    bool found = false;
    for (auto& p : roster_) {
        if (p.slot == info.slot) {
            p = info;
            found = true;
            break;
        }
    }
    if (!found) roster_.push_back(info);
    auto payload = encode_players(roster_, next_seq());
    net_.push(frame_packet(payload));
}

void PluginRuntime::on_player_leave(int slot) {
    if (!active_) return;
    roster_.erase(
        std::remove_if(roster_.begin(), roster_.end(),
                       [slot](const PlayerInfoFrame& p) { return p.slot == slot; }),
        roster_.end());
    auto payload = encode_players(roster_, next_seq());
    net_.push(frame_packet(payload));
}

void PluginRuntime::on_game_frame(double t, int tick,
                                  const std::vector<PlayerState>& players) {
    if (!active_) return;
    auto payload = encode_tick(t, tick, players, next_seq());
    net_.push(frame_packet(payload));

    uint32_t s = next_seq();
    vis_.tick(t, tick, net_, players, s);
    // vis_.tick may consume multiple seq numbers; keep atomic roughly aligned.
    seq_.store(s, std::memory_order_relaxed);
}

void PluginRuntime::on_weapon_fire(double t, int tick, int slot, int weapon_defindex,
                                   float pitch, float yaw,
                                   float punch_pitch, float punch_yaw,
                                   float spread) {
    if (!active_) return;
    // CRITICAL: pitch/yaw must be the server fire angles; punch_* is
    // m_aimPunchAngle accumulated, not a per-shot delta.
    auto payload = encode_shot(t, tick, slot, weapon_defindex, pitch, yaw,
                               punch_pitch, punch_yaw, spread, next_seq());
    net_.push(frame_packet(payload));
}

void PluginRuntime::on_player_hurt(double t, int tick, int attacker, int victim,
                                   float damage, int hitgroup, int weapon,
                                   float dist_units, bool victim_visible) {
    if (!active_) return;
    auto payload = encode_dmg(t, tick, attacker, victim, damage, hitgroup, weapon,
                              dist_units, victim_visible, next_seq());
    net_.push(frame_packet(payload));
}

void PluginRuntime::on_player_death(double t, int tick, int attacker, int victim,
                                    int weapon, bool hs, float dist_units,
                                    bool smoke, bool wall, bool flash, bool noscope,
                                    float exposure_ms) {
    if (!active_) return;
    auto payload = encode_kill(t, tick, attacker, victim, weapon, hs, dist_units,
                               smoke, wall, flash, noscope, exposure_ms, next_seq());
    net_.push(frame_packet(payload));
}

}  // namespace aegis

// ---------------------------------------------------------------------------
// MetaMod:Source2 entry — compiled only when AEGIS_WITH_METAMOD is defined.
// ---------------------------------------------------------------------------
#ifdef AEGIS_WITH_METAMOD

// These headers come from metamod-source / hl2sdk-cs2. Exact names shift across
// MMS versions; adjust includes when you point CMake at your SDK trees.
#  include <ISmmPlugin.h>
#  include <igameevents.h>
#  include <sh_vector.h>

class AegisPlugin : public ISmmPlugin, public IMetamodListener {
public:
    bool Load(PluginId id, ISmmAPI* ismm, char* error, size_t maxlen, bool late) override;
    bool Unload(char* error, size_t maxlen) override;
    const char* GetAuthor() override { return "Aegis"; }
    const char* GetName() override { return "Aegis CS2 Telemetry"; }
    const char* GetDescription() override {
        return "Streams canonical AC telemetry to aegis ingest";
    }
    const char* GetURL() override { return "https://github.com/p3nicillin/aegis"; }
    const char* GetLicense() override { return "Proprietary"; }
    const char* GetVersion() override { return "0.1.0"; }
    const char* GetDate() override { return __DATE__; }
    const char* GetLogTag() override { return "AEGIS"; }

private:
    aegis::PluginRuntime runtime_;
};

static AegisPlugin g_plugin;
PLUGIN_EXPOSE(AegisPlugin, g_plugin);

bool AegisPlugin::Load(PluginId id, ISmmAPI* ismm, char* error, size_t maxlen, bool late) {
    PLUGIN_SAVEVARS();
    (void)id; (void)ismm; (void)late;
    aegis::Config cfg;
    // TODO: read ConVars aegis_ingest_host / aegis_ingest_port / aegis_account_salt
    if (cfg.account_salt == "CHANGE_ME_SERVER_SALT") {
        // Refuse to ship unsalted ids.
        std::snprintf(error, maxlen, "set aegis_account_salt before loading");
        // Still load in scaffold mode so servers can bring the DLL up offline.
    }
    runtime_.load(cfg);

    // TODO(sdk): hook GameFrame, player_hurt, player_death, weapon_fire,
    //            CCSPlayerPawn m_angEyeAngles + m_aimPunchAngle on fire.
    // Example (pseudocode — signatures vary by SDK build):
    //   SH_ADD_HOOK(IServerGameDLL, GameFrame, gamedll, SH_MEMBER(this, &AegisPlugin::GameFrame), true);
    return true;
}

bool AegisPlugin::Unload(char* error, size_t maxlen) {
    (void)error; (void)maxlen;
    runtime_.unload();
    return true;
}

#endif  // AEGIS_WITH_METAMOD
