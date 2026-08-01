#pragma once

#include "frames.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace aegis {

struct SdkHooks {
    std::function<std::vector<PlayerState>()> collect_players;
    std::function<std::string()> current_map;
    std::function<float()> tickrate;
    std::function<bool(const PlayerState&, const PlayerState&)> trace_visible;
    std::function<bool()> vis_ready;
    std::function<uint64_t(int slot)> steamid64_for_slot;
};

SdkHooks& sdk_hooks();

namespace flag {
constexpr int ON_GROUND = 1 << 0;
constexpr int DUCKING = 1 << 1;
constexpr int SCOPED = 1 << 2;
constexpr int RELOADING = 1 << 3;
constexpr int DEFUSING = 1 << 4;
constexpr int PLANTING = 1 << 5;
constexpr int FLASHED = 1 << 6;
constexpr int SMOKED = 1 << 7;
constexpr int WALKING = 1 << 8;
constexpr int LADDER = 1 << 9;
constexpr int FROZEN = 1 << 10;
constexpr int BLIND_PARTIAL = 1 << 11;
}  // namespace flag

namespace btn {
constexpr int ATTACK = 1 << 0;
constexpr int JUMP = 1 << 1;
constexpr int DUCK = 1 << 2;
constexpr int FORWARD = 1 << 3;
constexpr int BACK = 1 << 4;
constexpr int USE = 1 << 5;
constexpr int MOVELEFT = 1 << 9;
constexpr int MOVERIGHT = 1 << 10;
}  // namespace btn

namespace sdk {

struct FireSnapshot {
    float pitch = 0, yaw = 0;
    float punch_pitch = 0, punch_yaw = 0;
    int weapon_defindex = 0;
    float spread = 0;
};

struct QAngle3 {
    float x = 0, y = 0, z = 0;  // pitch, yaw, roll — matches QAngle layout
    float pitch() const { return x; }
    float yaw() const { return y; }
};

struct Vec3 {
    float x = 0, y = 0, z = 0;
};

void install_sdk_hooks();
void mark_vis_ready(bool ok);
void set_engine_context(void* engine, void* server, void* server_config,
                        void* network_service, float tick_interval);

void cache_pawn(int slot, void* pawn);
void clear_pawn(int slot);
void* pawn_for_slot(int slot);
void set_last_weapon(int slot, int defindex);
int last_weapon(int slot);

PlayerState snapshot_player(int slot, void* pawn);
FireSnapshot snapshot_fire(void* pawn);
int weapon_defindex_from_name(const char* name);
float dist_between_slots(int a, int b);

}  // namespace sdk
}  // namespace aegis
