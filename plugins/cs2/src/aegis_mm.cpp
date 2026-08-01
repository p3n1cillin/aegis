#include "aegis_mm.h"

#include "frames.h"
#include "salt.h"
#include "sdk_bridge.h"

#include <algorithm>
#include <array>
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

#  include "module_scan.h"
#  include "schema_offset.h"

#  include <ISmmPlugin.h>
#  include <eiface.h>
#  include <igameevents.h>
#  include <iserver.h>
#  include <schemasystem/schemasystem.h>
#  include <sh_vector.h>

#  include <cstdarg>
#  include <cstdlib>
#  include <cstring>
#  include <string>

PLUGIN_GLOBALVARS();

SH_DECL_HOOK3_void(IServerGameDLL, GameFrame, SH_NOATTRIB, 0, bool, bool, bool);
SH_DECL_HOOK4_void(IServerGameClients, ClientActive, SH_NOATTRIB, 0, CPlayerSlot, bool,
                   const char*, uint64);
SH_DECL_HOOK5_void(IServerGameClients, ClientDisconnect, SH_NOATTRIB, 0, CPlayerSlot,
                   ENetworkDisconnectionReason, const char*, uint64, const char*);
SH_DECL_HOOK6_void(IServerGameClients, OnClientConnected, SH_NOATTRIB, 0, CPlayerSlot,
                   const char*, uint64, const char*, const char*, bool);
SH_DECL_HOOK2(IGameEventManager2, LoadEventsFromFile, SH_NOATTRIB, 0, int, const char*,
              bool);
SH_DECL_HOOK2(IGameEventManager2, FireEvent, SH_NOATTRIB, 0, bool, IGameEvent*, bool);

// MetaMod's console output never reaches a file, so lifecycle tracing goes to
// AEGIS_LOG_FILE when set. Cheap and off unless asked for.
static void aegis_log(const char* fmt, ...) {
    static const char* path = std::getenv("AEGIS_LOG_FILE");
    if (!path) return;
    FILE* f = nullptr;
    if (fopen_s(&f, path, "a") != 0 || !f) return;
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(f, fmt, ap);
    va_end(ap);
    std::fputc('\n', f);
    std::fclose(f);
}

static IServerGameDLL* g_pServer = nullptr;
static IServerGameClients* g_pGameClients = nullptr;
static IVEngineServer* g_pEngine = nullptr;
static IGameEventManager2* g_pGameEvents = nullptr;
static ISchemaSystem* g_pSchemaSystem = nullptr;
static int g_loadEventsHookId = -1;
static int g_fireEventHookId = -1;

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

    void OnLevelInit(char const* pMapName, char const* pMapEntities,
                     char const* pOldLevel, char const* pLandmarkName, bool loadGame,
                     bool background) override;
    void OnLevelShutdown() override;

    void Hook_GameFrame(bool simulating, bool bFirstTick, bool bLastTick);
    void Hook_ClientActive(CPlayerSlot slot, bool bLoadGame, const char* pszName,
                           uint64 xuid);
    void Hook_ClientDisconnect(CPlayerSlot slot, ENetworkDisconnectionReason reason,
                               const char* pszName, uint64 xuid,
                               const char* pszNetworkID);
    void Hook_OnClientConnected(CPlayerSlot slot, const char* pszName, uint64 xuid,
                                const char* pszNetworkID, const char* pszAddress,
                                bool bFakePlayer);
    int Hook_LoadEventsFromFile(const char* filename, bool bSearchAll);
    bool Hook_FireEvent(IGameEvent* event, bool bDontBroadcast);

    void FireGameEvent(IGameEvent* event) override;

private:
    void read_config(aegis::Config& cfg);
    void register_events();
    void unregister_events();
    void ensure_map_session();
    double now_t() const;
    int now_tick() const;

    aegis::PluginRuntime runtime_;
    bool disabled_ = false;
    bool map_active_ = false;
    bool events_registered_ = false;
    std::array<bool, 64> bot_slot_{};
    std::array<std::string, 64> name_slot_{};
};

static AegisPlugin g_plugin;
PLUGIN_EXPOSE(AegisPlugin, g_plugin);

void AegisPlugin::read_config(aegis::Config& cfg) {
    if (const char* h = std::getenv("AEGIS_INGEST_HOST")) cfg.ingest_host = h;
    if (const char* p = std::getenv("AEGIS_INGEST_PORT")) cfg.ingest_port = std::atoi(p);
    if (const char* s = std::getenv("AEGIS_ACCOUNT_SALT")) cfg.account_salt = s;
    if (const char* v = std::getenv("AEGIS_EMIT_VIS")) cfg.emit_vis = std::atoi(v) != 0;
}

bool AegisPlugin::Load(PluginId id, ISmmAPI* ismm, char* error, size_t maxlen, bool late) {
    PLUGIN_SAVEVARS();
    (void)id;
    (void)late;

    // Escape hatch: load inert so the server can boot without Aegis touching
    // any engine state (used to isolate plugin-caused faults).
    if (const char* off = std::getenv("AEGIS_DISABLE"); off && std::atoi(off) != 0) {
        disabled_ = true;
        META_CONPRINTF("[Aegis] AEGIS_DISABLE set — loaded inert\n");
        return true;
    }

    GET_V_IFACE_CURRENT(GetEngineFactory, g_pEngine, IVEngineServer,
                        INTERFACEVERSION_VENGINESERVER);
    GET_V_IFACE_ANY(GetServerFactory, g_pServer, IServerGameDLL,
                    INTERFACEVERSION_SERVERGAMEDLL);
    GET_V_IFACE_ANY(GetServerFactory, g_pGameClients, IServerGameClients,
                    INTERFACEVERSION_SERVERGAMECLIENTS);
    // Do NOT request ISource2ServerConfig here. MetaMod patches that interface's
    // vtable while the game DLL is loading, and fetching it through the server
    // factory during Load leaves the engine calling a clobbered slot (AV on
    // startup). Tickrate is derived from CGlobalVars instead.
    GET_V_IFACE_ANY(GetEngineFactory, g_pNetworkServerService, INetworkServerService,
                    NETWORKSERVERSERVICE_INTERFACE_VERSION);
    GET_V_IFACE_CURRENT(GetEngineFactory, g_pSchemaSystem, ISchemaSystem,
                        SCHEMASYSTEM_INTERFACE_VERSION);

    if (!g_pServer || !g_pEngine) {
        std::snprintf(error, maxlen, "Aegis: failed to resolve server/engine interfaces");
        return false;
    }

    g_SMAPI->AddListener(this, this);

    aegis::schema::init(g_pSchemaSystem);

    aegis::sdk::set_engine_context(g_pEngine, g_pServer, nullptr,
                                   g_pNetworkServerService, 1.0f / 64.0f);
    aegis::sdk::install_sdk_hooks();

    aegis::Config cfg;
    read_config(cfg);
    if (cfg.account_salt.empty() || cfg.account_salt == "CHANGE_ME_SERVER_SALT") {
        META_CONPRINTF("[Aegis] WARNING: set AEGIS_ACCOUNT_SALT for production\n");
    }
    runtime_.load(cfg);

    SH_ADD_HOOK(IServerGameDLL, GameFrame, g_pServer,
                SH_MEMBER(this, &AegisPlugin::Hook_GameFrame), true);
    SH_ADD_HOOK(IServerGameClients, ClientActive, g_pGameClients,
                SH_MEMBER(this, &AegisPlugin::Hook_ClientActive), true);
    SH_ADD_HOOK(IServerGameClients, ClientDisconnect, g_pGameClients,
                SH_MEMBER(this, &AegisPlugin::Hook_ClientDisconnect), true);
    SH_ADD_HOOK(IServerGameClients, OnClientConnected, g_pGameClients,
                SH_MEMBER(this, &AegisPlugin::Hook_OnClientConnected), true);

    // Capture IGameEventManager2* the first time the engine loads event files.
    // The engine exposes no interface version for it, so the pointer has to come
    // from an RTTI-derived vtable in the game server module. That scan patches
    // engine memory, so it is opt-in: AEGIS_EVENT_HOOK=1.
    // Resolve via the server interface address — MetaMod also ships a stub named
    // server.dll that GetModuleHandle("server.dll") would hit first.
    const char* want_events = std::getenv("AEGIS_EVENT_HOOK");
    if (want_events && std::atoi(want_events) != 0) {
        aegis::mod::Module server_mod = aegis::mod::Module::from_address(g_pServer);
        META_CONPRINTF("[Aegis] server module base=%p size=%llu\n", server_mod.base(),
                       static_cast<unsigned long long>(server_mod.size()));
        if (void* vt = server_mod.find_virtual_table("CGameEventManager")) {
            g_loadEventsHookId = SH_ADD_DVPHOOK(
                IGameEventManager2, LoadEventsFromFile,
                static_cast<IGameEventManager2*>(vt),
                SH_MEMBER(this, &AegisPlugin::Hook_LoadEventsFromFile), false);
            // Intercept dispatch directly. AddListener() reports success on CS2 but
            // never delivers to a plugin listener, whereas every server-side event
            // passes through FireEvent.
            g_fireEventHookId = SH_ADD_DVPHOOK(
                IGameEventManager2, FireEvent, static_cast<IGameEventManager2*>(vt),
                SH_MEMBER(this, &AegisPlugin::Hook_FireEvent), false);
            META_CONPRINTF("[Aegis] hooked CGameEventManager::LoadEventsFromFile vt=%p\n",
                           vt);
            aegis_log("event hooks installed vt=%p load_id=%d fire_id=%d", vt,
                      g_loadEventsHookId, g_fireEventHookId);
        } else {
            META_CONPRINTF("[Aegis] WARN: CGameEventManager vtable not found — "
                           "shot/hurt/kill frames disabled\n");
            aegis_log("WARN CGameEventManager vtable not found (base=%p size=%llu)",
                      server_mod.base(),
                      static_cast<unsigned long long>(server_mod.size()));
        }
    } else {
        META_CONPRINTF("[Aegis] event hook disabled (set AEGIS_EVENT_HOOK=1 for "
                       "shot/hurt/kill frames)\n");
    }

    if (late) {
        ensure_map_session();
        register_events();
    }

    META_CONPRINTF("[Aegis] loaded — ingest %s:%d\n", cfg.ingest_host.c_str(),
                   cfg.ingest_port);
    aegis_log("loaded ingest=%s:%d late=%d", cfg.ingest_host.c_str(), cfg.ingest_port,
              late ? 1 : 0);
    return true;
}

bool AegisPlugin::Unload(char* error, size_t maxlen) {
    (void)error;
    (void)maxlen;
    if (disabled_) return true;
    unregister_events();
    if (g_loadEventsHookId >= 0) {
        SH_REMOVE_HOOK_ID(g_loadEventsHookId);
        g_loadEventsHookId = -1;
    }
    if (g_fireEventHookId >= 0) {
        SH_REMOVE_HOOK_ID(g_fireEventHookId);
        g_fireEventHookId = -1;
    }
    SH_REMOVE_HOOK(IServerGameDLL, GameFrame, g_pServer,
                   SH_MEMBER(this, &AegisPlugin::Hook_GameFrame), true);
    SH_REMOVE_HOOK(IServerGameClients, ClientActive, g_pGameClients,
                   SH_MEMBER(this, &AegisPlugin::Hook_ClientActive), true);
    SH_REMOVE_HOOK(IServerGameClients, ClientDisconnect, g_pGameClients,
                   SH_MEMBER(this, &AegisPlugin::Hook_ClientDisconnect), true);
    SH_REMOVE_HOOK(IServerGameClients, OnClientConnected, g_pGameClients,
                   SH_MEMBER(this, &AegisPlugin::Hook_OnClientConnected), true);
    runtime_.unload();
    return true;
}

void AegisPlugin::register_events() {
    // The FireEvent hook already sees every event; registering a listener too
    // would double-count anything it did deliver.
    if (g_fireEventHookId >= 0) return;
    if (!g_pGameEvents || events_registered_) {
        aegis_log("register_events skipped mgr=%p already=%d",
                  static_cast<void*>(g_pGameEvents), events_registered_ ? 1 : 0);
        return;
    }
    static const char* kEvents[] = {
        "round_start", "weapon_fire", "player_hurt", "player_death",
        "player_spawn", "player_connect_full", "player_disconnect",
    };
    int ok = 0;
    for (const char* name : kEvents) {
        if (g_pGameEvents->AddListener(this, name, true)) ++ok;
        else aegis_log("AddListener FAILED for %s", name);
    }
    events_registered_ = true;
    META_CONPRINTF("[Aegis] game event listeners registered\n");
    aegis_log("register_events mgr=%p ok=%d/%d", static_cast<void*>(g_pGameEvents), ok,
              static_cast<int>(sizeof(kEvents) / sizeof(kEvents[0])));
}

void AegisPlugin::unregister_events() {
    if (g_pGameEvents && events_registered_) {
        g_pGameEvents->RemoveListener(this);
    }
    events_registered_ = false;
}

bool AegisPlugin::Hook_FireEvent(IGameEvent* event, bool /*bDontBroadcast*/) {
    if (event) FireGameEvent(event);
    RETURN_META_VALUE(MRES_IGNORED, true);
}

int AegisPlugin::Hook_LoadEventsFromFile(const char* filename, bool bSearchAll) {
    (void)filename;
    (void)bSearchAll;
    if (!g_pGameEvents) {
        g_pGameEvents = META_IFACEPTR(IGameEventManager2);
        META_CONPRINTF("[Aegis] captured IGameEventManager2\n");
        aegis_log("captured IGameEventManager2 mgr=%p file=%s",
                  static_cast<void*>(g_pGameEvents), filename ? filename : "?");
        register_events();
    }
    RETURN_META_VALUE(MRES_IGNORED, 0);
}

void AegisPlugin::ensure_map_session() {
    if (map_active_) return;
    auto& h = aegis::sdk_hooks();
    const std::string map = h.current_map ? h.current_map() : std::string{};
    const float tr = h.tickrate ? h.tickrate() : 64.0f;
    aegis::sdk::mark_vis_ready(false);  // TraceShape unavailable in public SDK
    runtime_.on_map_start(map.empty() ? "unknown" : map, tr);
    map_active_ = true;
    META_CONPRINTF("[Aegis] session start map=%s tickrate=%.1f\n",
                   map.empty() ? "unknown" : map.c_str(), tr);
    aegis_log("session start map=%s tickrate=%.1f",
              map.empty() ? "unknown" : map.c_str(), tr);
}

void AegisPlugin::OnLevelInit(char const* pMapName, char const*, char const*,
                              char const*, bool, bool) {
    map_active_ = false;
    auto& h = aegis::sdk_hooks();
    const float tr = h.tickrate ? h.tickrate() : 64.0f;
    aegis::sdk::mark_vis_ready(false);
    runtime_.on_map_start(pMapName ? pMapName : "unknown", tr);
    map_active_ = true;
    register_events();
    META_CONPRINTF("[Aegis] OnLevelInit(%s)\n", pMapName ? pMapName : "?");
    aegis_log("OnLevelInit map=%s tickrate=%.1f", pMapName ? pMapName : "?", tr);
}

void AegisPlugin::OnLevelShutdown() {
    runtime_.on_map_end();
    map_active_ = false;
    for (int i = 0; i < 64; ++i) aegis::sdk::clear_pawn(i);
    META_CONPRINTF("[Aegis] OnLevelShutdown\n");
}

double AegisPlugin::now_t() const {
    if (g_pNetworkServerService && g_pNetworkServerService->GetIGameServer()) {
        if (CGlobalVars* g = g_pNetworkServerService->GetIGameServer()->GetGlobals()) {
            return g->curtime;
        }
    }
    return 0.0;
}

int AegisPlugin::now_tick() const {
    if (g_pNetworkServerService && g_pNetworkServerService->GetIGameServer()) {
        if (CGlobalVars* g = g_pNetworkServerService->GetIGameServer()->GetGlobals()) {
            return g->tickcount;
        }
    }
    return 0;
}

void AegisPlugin::Hook_GameFrame(bool simulating, bool /*bFirstTick*/,
                                 bool /*bLastTick*/) {
    if (!simulating) return;
    ensure_map_session();
    if (!map_active_) return;
    // Throttle tick frames to ~16 Hz to keep ingest light; shots/hurt stay event-driven.
    static int last_emit_tick = -1000;
    const int tick = now_tick();
    if (tick - last_emit_tick < 4) return;
    last_emit_tick = tick;
    runtime_.tick_from_sdk(now_t(), tick);

    static int emitted = 0;
    if (++emitted <= 3 || emitted % 200 == 0) {
        auto& h = aegis::sdk_hooks();
        const auto ps = h.collect_players ? h.collect_players()
                                          : std::vector<aegis::PlayerState>{};
        if (ps.empty()) {
            aegis_log("tick emit #%d tick=%d players=0", emitted, tick);
        } else {
            const auto& p = ps.front();
            aegis_log("tick emit #%d tick=%d players=%llu p0 pos=(%.1f,%.1f,%.1f) "
                      "ang=(%.1f,%.1f) hp=%d w=%d",
                      emitted, tick, static_cast<unsigned long long>(ps.size()), p.x, p.y,
                      p.z, p.pitch, p.yaw, p.hp, p.weapon_defindex);
        }
    }
}

void AegisPlugin::Hook_OnClientConnected(CPlayerSlot slot, const char* pszName,
                                         uint64 /*xuid*/, const char* /*pszNetworkID*/,
                                         const char* /*pszAddress*/, bool bFakePlayer) {
    const int s = slot.Get();
    if (s < 0 || s >= 64) return;
    bot_slot_[static_cast<size_t>(s)] = bFakePlayer;
    name_slot_[static_cast<size_t>(s)] = pszName ? pszName : "";
}

void AegisPlugin::Hook_ClientActive(CPlayerSlot slot, bool /*bLoadGame*/,
                                    const char* pszName, uint64 /*xuid*/) {
    ensure_map_session();
    const int s = slot.Get();
    if (s < 0 || s >= 64) return;
    if (pszName && *pszName) name_slot_[static_cast<size_t>(s)] = pszName;
    const bool bot = bot_slot_[static_cast<size_t>(s)];
    auto info = runtime_.make_roster_entry(
        s, name_slot_[static_cast<size_t>(s)], 0, bot);
    runtime_.on_player_join(info);
    aegis_log("ClientActive slot=%d name=%s bot=%d", s,
              name_slot_[static_cast<size_t>(s)].c_str(), bot ? 1 : 0);
}

void AegisPlugin::Hook_ClientDisconnect(CPlayerSlot slot,
                                        ENetworkDisconnectionReason /*reason*/,
                                        const char* /*pszName*/, uint64 /*xuid*/,
                                        const char* /*pszNetworkID*/) {
    const int s = slot.Get();
    if (s < 0 || s >= 64) return;
    aegis::sdk::clear_pawn(s);
    bot_slot_[static_cast<size_t>(s)] = false;
    name_slot_[static_cast<size_t>(s)].clear();
    runtime_.on_player_leave(s);
}

void AegisPlugin::FireGameEvent(IGameEvent* event) {
    if (!event) return;
    const char* name = event->GetName();
    if (!name) return;

    ensure_map_session();
    const double t = now_t();
    const int tick = now_tick();

    static int n_any = 0;
    if (++n_any <= 12) aegis_log("event[%d] %s", n_any, name);

    static int n_fire = 0, n_hurt = 0, n_kill = 0;
    if (std::strcmp(name, "weapon_fire") == 0) ++n_fire;
    else if (std::strcmp(name, "player_hurt") == 0) ++n_hurt;
    else if (std::strcmp(name, "player_death") == 0) ++n_kill;
    if ((n_fire + n_hurt + n_kill) % 25 == 1) {
        aegis_log("events fire=%d hurt=%d kill=%d", n_fire, n_hurt, n_kill);
    }

    if (std::strcmp(name, "round_start") == 0) {
        return;
    }

    if (std::strcmp(name, "player_spawn") == 0 ||
        std::strcmp(name, "player_connect_full") == 0) {
        const CPlayerSlot slot = event->GetPlayerSlot("userid");
        const int s = slot.Get();
        if (s < 0 || s >= 64) return;
        if (void* pawn = event->GetPlayerPawn("userid")) {
            aegis::sdk::cache_pawn(s, pawn);
        }
        if (std::strcmp(name, "player_connect_full") == 0) {
            const char* n = event->GetString("name",
                                             name_slot_[static_cast<size_t>(s)].c_str());
            auto info = runtime_.make_roster_entry(
                s, n ? n : "", 0, bot_slot_[static_cast<size_t>(s)]);
            runtime_.on_player_join(info);
        }
        return;
    }

    if (std::strcmp(name, "player_disconnect") == 0) {
        const int s = event->GetPlayerSlot("userid").Get();
        if (s >= 0 && s < 64) {
            aegis::sdk::clear_pawn(s);
            runtime_.on_player_leave(s);
        }
        return;
    }

    if (std::strcmp(name, "weapon_fire") == 0) {
        const int s = event->GetPlayerSlot("userid").Get();
        void* pawn = event->GetPlayerPawn("userid");
        if (pawn) aegis::sdk::cache_pawn(s, pawn);
        auto fs = aegis::sdk::snapshot_fire(pawn);
        const char* wname = event->GetString("weapon", "");
        const int def = aegis::sdk::weapon_defindex_from_name(wname);
        fs.weapon_defindex = def;
        aegis::sdk::set_last_weapon(s, def);
        runtime_.on_weapon_fire(t, tick, s, def, fs.pitch, fs.yaw, fs.punch_pitch,
                                fs.punch_yaw, fs.spread);
        return;
    }

    if (std::strcmp(name, "player_hurt") == 0) {
        const int victim = event->GetPlayerSlot("userid").Get();
        const int attacker = event->GetPlayerSlot("attacker").Get();
        if (void* vp = event->GetPlayerPawn("userid")) aegis::sdk::cache_pawn(victim, vp);
        if (void* ap = event->GetPlayerPawn("attacker"))
            aegis::sdk::cache_pawn(attacker, ap);
        const float dmg = event->GetFloat("dmg_health", 0.f);
        const int hitgroup = event->GetInt("hitgroup", 0);
        const char* wname = event->GetString("weapon", "");
        int def = aegis::sdk::weapon_defindex_from_name(wname);
        if (!def) def = aegis::sdk::last_weapon(attacker);
        const float dist = aegis::sdk::dist_between_slots(attacker, victim);
        // Visibility unknown without TraceShape — report false, never invent.
        runtime_.on_player_hurt(t, tick, attacker, victim, dmg, hitgroup, def, dist,
                                false);
        return;
    }

    if (std::strcmp(name, "player_death") == 0) {
        const int victim = event->GetPlayerSlot("userid").Get();
        const int attacker = event->GetPlayerSlot("attacker").Get();
        if (void* vp = event->GetPlayerPawn("userid")) aegis::sdk::cache_pawn(victim, vp);
        if (void* ap = event->GetPlayerPawn("attacker"))
            aegis::sdk::cache_pawn(attacker, ap);
        const char* wname = event->GetString("weapon", "");
        int def = aegis::sdk::weapon_defindex_from_name(wname);
        if (!def) def = aegis::sdk::last_weapon(attacker);
        const bool hs = event->GetBool("headshot", false);
        const bool noscope = event->GetBool("noscope", false);
        const bool smoke = event->GetBool("thrusmoke", false);
        const bool wall = event->GetInt("penetrated", 0) > 0;
        const bool flash = event->GetBool("attackerblind", false);
        const float dist = aegis::sdk::dist_between_slots(attacker, victim);
        runtime_.on_player_death(t, tick, attacker, victim, def, hs, dist, smoke, wall,
                                 flash, noscope, -1.f);
        return;
    }
}

#endif  // AEGIS_WITH_METAMOD
