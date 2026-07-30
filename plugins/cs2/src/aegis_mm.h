#pragma once

#include "config.h"
#include "frames.h"
#include "net_client.h"
#include "vis_worker.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace aegis {

// Plugin runtime shared by the MetaMod entry and the SDK-free harness.
class PluginRuntime {
public:
    void load(const Config& cfg);
    void unload();

    void on_map_start(const std::string& map, float tickrate);
    void on_map_end();
    void on_player_join(const PlayerInfoFrame& info);
    void on_player_leave(int slot);

    // Called every server tick — must stay cheap; only snapshots + enqueue.
    void on_game_frame(double t, int tick, const std::vector<PlayerState>& players);

    // Prefer this from MetaMod GameFrame — pulls players via SdkHooks.
    void tick_from_sdk(double t, int tick);

    // Fire with the view angle the server used for the shot and the accumulated
    // m_aimPunchAngle — never the post-shot client angle.
    void on_weapon_fire(double t, int tick, int slot, int weapon_defindex,
                        float pitch, float yaw,
                        float punch_pitch, float punch_yaw, float spread);

    void on_player_hurt(double t, int tick, int attacker, int victim,
                        float damage, int hitgroup, int weapon,
                        float dist_units, bool victim_visible);

    void on_player_death(double t, int tick, int attacker, int victim,
                         int weapon, bool hs, float dist_units,
                         bool smoke, bool wall, bool flash, bool noscope,
                         float exposure_ms);

    PlayerInfoFrame make_roster_entry(int slot, const std::string& name,
                                      int team, bool bot,
                                      float trust = 0.5f, int rank = 0,
                                      float hours = 0);

    NetClient& net() { return net_; }
    uint32_t seq() const { return seq_; }
    const Config& config() const { return cfg_; }

private:
    Config cfg_;
    NetClient net_;
    VisWorker vis_;
    std::string session_id_;
    std::vector<PlayerInfoFrame> roster_;
    std::atomic<uint32_t> seq_{1};
    bool active_ = false;

    uint32_t next_seq() { return seq_++; }
};

}  // namespace aegis
