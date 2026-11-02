#pragma once

#include "encode.h"

#include <cstdint>
#include <string>
#include <vector>

namespace aegis {

struct PlayerState {
    int slot = 0;
    float x = 0, y = 0, z = 0;
    float vx = 0, vy = 0, vz = 0;
    float pitch = 0, yaw = 0;
    int hp = 0, armor = 0;
    int weapon_defindex = 0;
    int ammo = 0;
    int flags = 0;
    int shots_fired = 0;
    int aim_target = -1;
    int aim_target_vis = 0;
    float move_fwd = 0, move_side = 0;
    int buttons = 0;
};

struct PlayerInfoFrame {
    int slot = 0;
    std::string pid;  // already salted
    std::string name;
    int team = 0;
    bool bot = false;
    float trust = 0.5f;
    int rank = 0;
    float hours = 0;
};

// Wire frames matching PLUGIN_PROTOCOL.md. Each returns a msgpack map payload
// (NOT length-delimited — NetClient wraps that).
std::vector<uint8_t> encode_hdr(const std::string& session_id,
                                 const std::string& map,
                                 float tickrate,
                                 const std::string& server,
                                 const std::string& ver,
                                 bool no_vis,
                                 uint32_t seq);

// First frame on every connection when an ingest token is configured.
std::vector<uint8_t> encode_auth(const std::string& token);

std::vector<uint8_t> encode_players(const std::vector<PlayerInfoFrame>& players,
                                    uint32_t seq);

std::vector<uint8_t> encode_tick(double t, int tick,
                                 const std::vector<PlayerState>& players,
                                 uint32_t seq);

std::vector<uint8_t> encode_shot(double t, int tick, int slot, int weapon,
                                 float pitch, float yaw,
                                 float recoil_pitch, float recoil_yaw,
                                 float spread, uint32_t seq);

std::vector<uint8_t> encode_vis(double t, int tick, int observer, int subject,
                                bool visible, float angle_error_deg,
                                float dist_units, uint32_t seq);

std::vector<uint8_t> encode_dmg(double t, int tick, int attacker, int victim,
                                float damage, int hitgroup, int weapon,
                                float dist_units, bool victim_visible,
                                uint32_t seq);

std::vector<uint8_t> encode_kill(double t, int tick, int attacker, int victim,
                                 int weapon, bool hs, float dist_units,
                                 bool smoke, bool wall, bool flash, bool noscope,
                                 float exp_ms, uint32_t seq);

// Length-delimited TCP frame: [flags:u8][len:u32 BE][payload]
// flags bit0 = zstd (payload already compressed by caller if set).
std::vector<uint8_t> frame_packet(const std::vector<uint8_t>& payload,
                                  uint8_t flags = 0);

}  // namespace aegis
