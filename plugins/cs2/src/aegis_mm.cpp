#include "aegis_mm.h"

#include "frames.h"
#include "salt.h"
#include "sdk_bridge.h"

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
    // Vis stays off until SdkHooks::vis_ready reports a working TraceShape.
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

    auto& hooks = sdk_hooks();
    const bool vis_ok = cfg_.emit_vis && hooks.vis_ready && hooks.vis_ready();
    vis_.set_enabled(vis_ok);
    if (hooks.trace_visible) {
        vis_.set_probe(hooks.trace_visible);
    }

    const bool no_vis = !vis_ok;
    auto hdr = encode_hdr(session_id_, map, tickrate, cfg_.server_name,
                          cfg_.plugin_version, no_vis, next_seq());
    net_.push(frame_packet(hdr));
}

void PluginRuntime::on_map_end() {
    active_ = false;
    roster_.clear();
    vis_.set_enabled(false);
}

void PluginRuntime::on_player_join(const PlayerInfoFrame& info) {
    if (!active_) return;
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
    seq_.store(s, std::memory_order_relaxed);
}

void PluginRuntime::tick_from_sdk(double t, int tick) {
    auto& hooks = sdk_hooks();
    if (!hooks.collect_players) return;
    on_game_frame(t, tick, hooks.collect_players());
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

PlayerInfoFrame PluginRuntime::make_roster_entry(int slot, const std::string& name,
                                                 int team, bool bot,
                                                 float trust, int rank, float hours) {
    PlayerInfoFrame info;
    info.slot = slot;
    info.name = name;
    info.team = team;
    info.bot = bot;
    info.trust = trust;
    info.rank = rank;
    info.hours = hours;
    uint64_t sid = 0;
    if (auto& h = sdk_hooks(); h.steamid64_for_slot) sid = h.steamid64_for_slot(slot);
    info.pid = salt_account_id(cfg_.account_salt, sid);
    return info;
}

}  // namespace aegis

#ifdef AEGIS_WITH_METAMOD

#  include <ISmmPlugin.h>
#  include <igameevents.h>
#  include <sh_vector.h>

#  include <cstdlib>
#  include <cstring>

namespace aegis {
namespace sdk {
void install_sdk_hooks();
void mark_vis_ready(bool ok);
}  // namespace sdk
}  // namespace aegis

class AegisPlugin : public ISmmPlugin,
                    public IMetamodListener,
                    public IGameEventListener2 {
public:
    bool Load(PluginId id, ISmmAPI* ismm, char* error, size_t maxlen, bool late) override;
    bool Unload(char* error, size_t maxlen) override;
    void AllPluginsLoaded() override {}

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

    // SourceHook targets -------------------------------------------------
    void Hook_GameFrame(bool simulating, bool bFirstTick, bool bLastTick);
    void Hook_GameEvent(IGameEvent* event, bool dontBroadcast);

    // IGameEventListener2
    void FireGameEvent(IGameEvent* event) override;

private:
    void read_config(aegis::Config& cfg);
    void register_events();
    void unregister_events();

    aegis::PluginRuntime runtime_;
    double t0_ = 0.0;
    int tick_ = 0;
    bool map_active_ = false;
};

static AegisPlugin g_plugin;
PLUGIN_EXPOSE(AegisPlugin, g_plugin);

// ConVar names (create with ConVar / CCommand in your SDK flavour).
//   aegis_ingest_host   default 127.0.0.1
//   aegis_ingest_port   default 9100
//   aegis_account_salt  REQUIRED — refuse unsalted in production
//   aegis_emit_vis      default 1 (still gated on TraceShape readiness)

void AegisPlugin::read_config(aegis::Config& cfg) {
    // TODO(convar): bind real ConVars. getenv keeps lab boots working.
    if (const char* h = std::getenv("AEGIS_INGEST_HOST")) cfg.ingest_host = h;
    if (const char* p = std::getenv("AEGIS_INGEST_PORT")) cfg.ingest_port = std::atoi(p);
    if (const char* s = std::getenv("AEGIS_ACCOUNT_SALT")) cfg.account_salt = s;
    if (const char* v = std::getenv("AEGIS_EMIT_VIS")) cfg.emit_vis = std::atoi(v) != 0;
}

bool AegisPlugin::Load(PluginId id, ISmmAPI* ismm, char* error, size_t maxlen, bool late) {
    PLUGIN_SAVEVARS();
    (void)id; (void)ismm; (void)late;

    aegis::Config cfg;
    read_config(cfg);
    if (cfg.account_salt.empty() || cfg.account_salt == "CHANGE_ME_SERVER_SALT") {
        std::snprintf(error, maxlen,
                      "Aegis: set AEGIS_ACCOUNT_SALT / aegis_account_salt before production use");
        // Load anyway so the DLL can be brought up offline for wiring.
    }

    aegis::sdk::install_sdk_hooks();
    runtime_.load(cfg);

    // TODO(sdk): resolve IServerGameDLL* and SH_ADD_HOOK GameFrame.
    //   SH_ADD_HOOK_MEMFUNC(IServerGameDLL, GameFrame, gamedll, this,
    //                       &AegisPlugin::Hook_GameFrame, true);

    // TODO(sdk): resolve IGameEventManager2* and AddListener for:
    //   round_start, round_end, player_hurt, player_death, weapon_fire,
    //   player_connect_full, player_disconnect, bomb_planted, bomb_defused
    register_events();
    return true;
}

bool AegisPlugin::Unload(char* error, size_t maxlen) {
    (void)error; (void)maxlen;
    unregister_events();
    // TODO(sdk): SH_REMOVE_HOOK GameFrame
    runtime_.unload();
    return true;
}

void AegisPlugin::register_events() {
    // TODO(sdk):
    //   gameevents->AddListener(this, "round_start", true);
    //   gameevents->AddListener(this, "player_hurt", true);
    //   gameevents->AddListener(this, "player_death", true);
    //   gameevents->AddListener(this, "weapon_fire", true);
    //   gameevents->AddListener(this, "player_connect_full", true);
    //   gameevents->AddListener(this, "player_disconnect", true);
}

void AegisPlugin::unregister_events() {
    // TODO(sdk): gameevents->RemoveListener(this);
}

void AegisPlugin::Hook_GameFrame(bool simulating, bool /*bFirstTick*/, bool /*bLastTick*/) {
    if (!simulating || !map_active_) return;
    // gpGlobals->curtime / tickcount — replace once SDK globals are linked.
    tick_++;
    const double t = tick_ / 64.0;  // TODO(engine): real tickrate
    runtime_.tick_from_sdk(t, tick_);
}

void AegisPlugin::FireGameEvent(IGameEvent* event) {
    Hook_GameEvent(event, false);
}

void AegisPlugin::Hook_GameEvent(IGameEvent* event, bool /*dontBroadcast*/) {
    if (!event) return;
    const char* name = event->GetName();
    if (!name) return;

    if (std::strcmp(name, "round_start") == 0 || std::strcmp(name, "game_newmap") == 0) {
        auto& h = aegis::sdk_hooks();
        const std::string map = h.current_map ? h.current_map() : std::string{};
        const float tr = h.tickrate ? h.tickrate() : 64.0f;
        // Self-test TraceShape once per map; only then enable vis.
        // TODO(sdk): fire a known clear LOS pair and call mark_vis_ready(true).
        aegis::sdk::mark_vis_ready(false);
        runtime_.on_map_start(map, tr);
        map_active_ = true;
        tick_ = 0;
        return;
    }

    if (std::strcmp(name, "round_end") == 0) {
        // Keep session open across rounds; only close on map change / unload.
        return;
    }

    if (std::strcmp(name, "weapon_fire") == 0) {
        // TODO(sdk):
        //   int slot = ... from userid
        //   auto* pawn = ...
        //   auto fs = aegis::sdk::snapshot_fire(pawn);
        //   runtime_.on_weapon_fire(t, tick_, slot, fs.weapon_defindex,
        //                           fs.pitch, fs.yaw, fs.punch_pitch, fs.punch_yaw, fs.spread);
        return;
    }

    if (std::strcmp(name, "player_hurt") == 0) {
        // TODO(sdk): map userid→slot, weapon defindex, hitgroup, distance,
        //            victim_visible from last vis state (not invent).
        return;
    }

    if (std::strcmp(name, "player_death") == 0) {
        // TODO(sdk): headshot / noscope / thrusmoke / thruhwall / flash /
        //            continuous visibility exposure ms.
        return;
    }

    if (std::strcmp(name, "player_connect_full") == 0) {
        // TODO(sdk): build PlayerInfoFrame via runtime_.make_roster_entry(...)
        return;
    }

    if (std::strcmp(name, "player_disconnect") == 0) {
        // TODO(sdk): runtime_.on_player_leave(slot);
        return;
    }
}

#endif  // AEGIS_WITH_METAMOD
