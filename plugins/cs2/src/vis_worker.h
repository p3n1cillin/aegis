#pragma once

#include "frames.h"
#include "net_client.h"

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aegis {

// 16 Hz occlusion worker. SDK-backed TraceShape goes in set_probe(); until
// then leave disabled and set hdr.no_vis — never invent LOS.
class VisWorker {
public:
    using ProbeFn = std::function<bool(const PlayerState&, const PlayerState&)>;

    void configure(float hz, float min_interval_s);
    void set_enabled(bool on) { enabled_ = on; }
    bool enabled() const { return enabled_; }
    void set_probe(ProbeFn fn) { probe_ = std::move(fn); }

    void tick(double t, int server_tick, NetClient& net,
              const std::vector<PlayerState>& players,
              uint32_t& seq);

    bool probe_pair(const PlayerState& observer, const PlayerState& subject) const;

private:
    float hz_ = 16.0f;
    float min_interval_s_ = 0.10f;
    bool enabled_ = false;
    double last_pass_t_ = -1.0;
    ProbeFn probe_;
    std::unordered_map<uint64_t, std::pair<bool, double>> last_;

    static uint64_t key(int o, int u) {
        return (uint64_t(uint32_t(o)) << 32) | uint32_t(u);
    }
};

}  // namespace aegis
