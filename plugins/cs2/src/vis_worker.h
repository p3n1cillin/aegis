#pragma once

#include "frames.h"
#include "net_client.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aegis {

// 16 Hz occlusion worker. SDK-backed TraceShape goes in probe_pair(); until
// then emit_vis stays false and hdr.no_vis is set — never invent LOS.
class VisWorker {
public:
    void configure(float hz, float min_interval_s);
    void set_enabled(bool on) { enabled_ = on; }
    bool enabled() const { return enabled_; }

    // Called from a background tick with live player origins / eye angles.
    // When SDK traces are wired, this emits encode_vis frames via net.
    void tick(double t, int server_tick, NetClient& net,
              const std::vector<PlayerState>& players,
              uint32_t& seq);

    // Hook point — return true if observer can see subject.
    // TODO(sdk): CGameTrace / TraceShape from eye to subject chest/head.
    bool probe_pair(const PlayerState& observer, const PlayerState& subject) const;

private:
    float hz_ = 16.0f;
    float min_interval_s_ = 0.10f;
    bool enabled_ = false;
    double last_pass_t_ = -1.0;
    std::unordered_map<uint64_t, std::pair<bool, double>> last_;  // key -> (vis, t)

    static uint64_t key(int o, int u) {
        return (uint64_t(uint32_t(o)) << 32) | uint32_t(u);
    }
};

}  // namespace aegis
