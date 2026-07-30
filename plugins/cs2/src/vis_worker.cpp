#include "vis_worker.h"

#include <cmath>

namespace aegis {

void VisWorker::configure(float hz, float min_interval_s) {
    hz_ = hz;
    min_interval_s_ = min_interval_s;
}

bool VisWorker::probe_pair(const PlayerState& /*observer*/,
                           const PlayerState& /*subject*/) const {
    // Intentionally unimplemented without SDK traces.
    return false;
}

void VisWorker::tick(double t, int server_tick, NetClient& net,
                     const std::vector<PlayerState>& players,
                     uint32_t& seq) {
    if (!enabled_) return;
    if (last_pass_t_ >= 0.0 && (t - last_pass_t_) < (1.0 / hz_)) return;
    last_pass_t_ = t;

    for (size_t i = 0; i < players.size(); ++i) {
        if (players[i].hp <= 0) continue;
        for (size_t j = 0; j < players.size(); ++j) {
            if (i == j || players[j].hp <= 0) continue;
            const int o = players[i].slot;
            const int u = players[j].slot;
            const bool vis = probe_pair(players[i], players[j]);
            const uint64_t k = key(o, u);
            auto it = last_.find(k);
            if (it != last_.end()) {
                if (it->second.first == vis) continue;
                if ((t - it->second.second) < min_interval_s_) continue;
            }
            last_[k] = {vis, t};

            const float dx = players[j].x - players[i].x;
            const float dy = players[j].y - players[i].y;
            const float dz = players[j].z - players[i].z;
            const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            // angle_error filled properly once eye vectors are available from SDK.
            auto payload = encode_vis(t, server_tick, o, u, vis, 0.0f, dist, seq++);
            net.push(frame_packet(payload));
        }
    }
}

}  // namespace aegis
