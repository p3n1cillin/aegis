#pragma once

#include <cstdint>
#include <string>

namespace aegis {

struct Config {
    std::string ingest_host = "127.0.0.1";
    int ingest_port = 9100;
    bool compress = true;
    bool emit_vis = true;          // set false / hdr.no_vis if traces unavailable
    float vis_hz = 16.0f;
    // Human motor tremor lives at 4-12 Hz; a 16 Hz telemetry rate (Nyquist
    // 8 Hz) cannot resolve most of that band, which silently invalidates the
    // spectral tremor channel on real matches even though it looks fine on
    // full-rate synthetic data. Bandwidth/CPU cost of the extra samples is
    // trivial for a server-side sidecar feed (10 players x ~15 fields at 64
    // Hz is still under 40 KB/s), so default to the engine's native rate and
    // only throttle down via AEGIS_TICK_HZ if an operator has a real reason.
    float tick_hz = 64.0f;
    float vis_pair_min_interval_s = 0.10f;
    std::string account_salt = "CHANGE_ME_SERVER_SALT";
    std::string server_name = "cs2";
    std::string plugin_version = "0.1.0";
};

}  // namespace aegis
