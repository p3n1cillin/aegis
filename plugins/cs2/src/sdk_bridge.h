#pragma once

#include "frames.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace aegis {

// Thin façade over hl2sdk / schema lookups. All engine-touching code lives
// behind these callbacks so the protocol runtime stays SDK-free.
//
// When AEGIS_WITH_METAMOD is off, the defaults return empty / false so the
// encode_smoke harness still links. When the SDK is present, sdk_bridge.cpp
// fills these from GameFrame + game events.

struct SdkHooks {
    // Snapshot every connected pawn into PlayerState (Hammer units, server
    // eye angles). Must be cheap — called every tick.
    std::function<std::vector<PlayerState>()> collect_players;

    // Map name + tickrate for hdr.
    std::function<std::string()> current_map;
    std::function<float()> tickrate;

    // TraceShape from observer eye to subject chest. Return false if the
    // engine cannot answer — caller must then leave vis disabled.
    std::function<bool(const PlayerState&, const PlayerState&)> trace_visible;

    // True once TraceShape is confirmed working this map.
    std::function<bool()> vis_ready;

    // Resolve slot -> SteamID64 for salting. 0 if unknown / bot.
    std::function<uint64_t(int slot)> steamid64_for_slot;
};

// Global hook table. MetaMod Load() installs; Unload() clears.
SdkHooks& sdk_hooks();

// Flag bit helpers matching PLUGIN_PROTOCOL.md (CS2 native ``f`` bitmask).
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

// Button bits matching adapter._BTN_MAP.
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

}  // namespace aegis
