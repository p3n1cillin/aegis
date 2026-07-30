#include "aegis_mm.h"
#include "frames.h"
#include "salt.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace aegis;

static bool expect_key(const std::vector<uint8_t>& b, const char* key) {
    // Naive search for fixstr key bytes in the payload — enough for a smoke test.
    const size_t n = std::strlen(key);
    if (n >= 32) return false;
    for (size_t i = 0; i + 1 + n <= b.size(); ++i) {
        if (b[i] == uint8_t(0xa0 | n) && std::memcmp(&b[i + 1], key, n) == 0) return true;
    }
    return false;
}

int main() {
    const auto pid = salt_account_id("lab-salt", 76561198000000000ull);
    if (pid.size() != 16) {
        std::fprintf(stderr, "salt length %zu\n", pid.size());
        return 1;
    }

    auto hdr = encode_hdr("sess-1", "de_anubis", 64.0f, "lab", "0.1.0", true, 1);
    auto pkt = frame_packet(hdr);
    if (pkt.size() < 5 || pkt[0] != 0) {
        std::fprintf(stderr, "bad packet header\n");
        return 1;
    }
    const uint32_t n = (uint32_t(pkt[1]) << 24) | (uint32_t(pkt[2]) << 16) |
                       (uint32_t(pkt[3]) << 8) | uint32_t(pkt[4]);
    if (n != hdr.size() || pkt.size() != 5 + n) {
        std::fprintf(stderr, "length mismatch %u vs %zu\n", n, hdr.size());
        return 1;
    }
    if (!expect_key(hdr, "hdr") || !expect_key(hdr, "no_vis") || !expect_key(hdr, "seq")) {
        std::fprintf(stderr, "hdr missing keys\n");
        return 1;
    }

    PlayerState p{};
    p.slot = 0;
    p.x = 100.f;
    p.y = 200.f;
    p.z = 64.f;
    p.hp = 100;
    p.weapon_defindex = 7;  // AK
    p.aim_target = -1;
    auto tick = encode_tick(0.015625, 1, {p}, 2);
    if (!expect_key(tick, "tick") || !expect_key(tick, "p")) {
        std::fprintf(stderr, "tick missing keys\n");
        return 1;
    }

    auto shot = encode_shot(0.1, 8, 0, 7, -1.2f, 45.0f, 2.5f, -0.4f, 0.01f, 3);
    if (!expect_key(shot, "rpi") || !expect_key(shot, "rya")) {
        std::fprintf(stderr, "shot missing recoil keys\n");
        return 1;
    }

    PluginRuntime rt;
    Config cfg;
    cfg.ingest_host = "127.0.0.1";
    cfg.ingest_port = 9;  // nothing listening — queue only
    cfg.emit_vis = false;
    rt.load(cfg);
    rt.on_map_start("de_anubis", 64.0f);
    PlayerInfoFrame info;
    info.slot = 0;
    info.pid = pid;
    info.name = "lab";
    info.team = 2;
    rt.on_player_join(info);
    rt.on_game_frame(0.0, 0, {p});
    rt.on_weapon_fire(0.1, 8, 0, 7, -1.2f, 45.0f, 2.5f, -0.4f, 0.01f);
    rt.unload();

    std::printf("encode_smoke ok pid=%s hdr=%zu tick=%zu shot=%zu packet=%zu\n",
                pid.c_str(), hdr.size(), tick.size(), shot.size(), pkt.size());
    return 0;
}
