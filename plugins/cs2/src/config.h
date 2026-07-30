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
    float vis_pair_min_interval_s = 0.10f;
    std::string account_salt = "CHANGE_ME_SERVER_SALT";
    std::string server_name = "cs2";
    std::string plugin_version = "0.1.0";
};

}  // namespace aegis
