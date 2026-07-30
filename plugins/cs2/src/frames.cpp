#include "frames.h"

namespace aegis {

static void put_seq(MsgpackWriter& w, uint32_t seq) {
    w.key("seq");
    w.u64(seq);
}

std::vector<uint8_t> encode_hdr(const std::string& session_id,
                                 const std::string& map,
                                 float tickrate,
                                 const std::string& server,
                                 const std::string& ver,
                                 bool no_vis,
                                 uint32_t seq) {
    MsgpackWriter w;
    w.map(no_vis ? 8 : 7);
    w.key("k"); w.str("hdr");
    w.key("session_id"); w.str(session_id);
    w.key("map"); w.str(map);
    w.key("tickrate"); w.f64(tickrate);
    w.key("server"); w.str(server);
    w.key("ver"); w.str(ver);
    if (no_vis) {
        w.key("no_vis"); w.boolean(true);
    }
    put_seq(w, seq);
    return w.bytes();
}

std::vector<uint8_t> encode_players(const std::vector<PlayerInfoFrame>& players,
                                    uint32_t seq) {
    MsgpackWriter w;
    w.map(3);
    w.key("k"); w.str("players");
    w.key("players");
    w.array(players.size());
    for (const auto& p : players) {
        w.map(8);
        w.key("slot"); w.i64(p.slot);
        w.key("pid"); w.str(p.pid);
        w.key("name"); w.str(p.name);
        w.key("team"); w.i64(p.team);
        w.key("bot"); w.boolean(p.bot);
        w.key("trust"); w.f64(p.trust);
        w.key("rank"); w.i64(p.rank);
        w.key("hours"); w.f64(p.hours);
    }
    put_seq(w, seq);
    return w.bytes();
}

std::vector<uint8_t> encode_tick(double t, int tick,
                                 const std::vector<PlayerState>& players,
                                 uint32_t seq) {
    MsgpackWriter w;
    w.map(5);
    w.key("k"); w.str("tick");
    w.key("t"); w.f64(t);
    w.key("tick"); w.i64(tick);
    w.key("p");
    w.array(players.size());
    for (const auto& p : players) {
        w.map(20);
        w.key("s"); w.i64(p.slot);
        w.key("x"); w.f64(p.x);
        w.key("y"); w.f64(p.y);
        w.key("z"); w.f64(p.z);
        w.key("vx"); w.f64(p.vx);
        w.key("vy"); w.f64(p.vy);
        w.key("vz"); w.f64(p.vz);
        w.key("pi"); w.f64(p.pitch);
        w.key("ya"); w.f64(p.yaw);
        w.key("hp"); w.i64(p.hp);
        w.key("ar"); w.i64(p.armor);
        w.key("w"); w.i64(p.weapon_defindex);
        w.key("am"); w.i64(p.ammo);
        w.key("f"); w.i64(p.flags);
        w.key("sf"); w.i64(p.shots_fired);
        w.key("at"); w.i64(p.aim_target);
        w.key("atv"); w.i64(p.aim_target_vis);
        w.key("mf"); w.f64(p.move_fwd);
        w.key("ms"); w.f64(p.move_side);
        w.key("b"); w.i64(p.buttons);
    }
    put_seq(w, seq);
    return w.bytes();
}

std::vector<uint8_t> encode_shot(double t, int tick, int slot, int weapon,
                                 float pitch, float yaw,
                                 float recoil_pitch, float recoil_yaw,
                                 float spread, uint32_t seq) {
    MsgpackWriter w;
    w.map(11);
    w.key("k"); w.str("shot");
    w.key("t"); w.f64(t);
    w.key("tick"); w.i64(tick);
    w.key("s"); w.i64(slot);
    w.key("w"); w.i64(weapon);
    w.key("pi"); w.f64(pitch);
    w.key("ya"); w.f64(yaw);
    w.key("rpi"); w.f64(recoil_pitch);
    w.key("rya"); w.f64(recoil_yaw);
    w.key("sp"); w.f64(spread);
    put_seq(w, seq);
    return w.bytes();
}

std::vector<uint8_t> encode_vis(double t, int tick, int observer, int subject,
                                bool visible, float angle_error_deg,
                                float dist_units, uint32_t seq) {
    MsgpackWriter w;
    w.map(9);
    w.key("k"); w.str("vis");
    w.key("t"); w.f64(t);
    w.key("tick"); w.i64(tick);
    w.key("o"); w.i64(observer);
    w.key("u"); w.i64(subject);
    w.key("v"); w.boolean(visible);
    w.key("ae"); w.f64(angle_error_deg);
    w.key("dist"); w.f64(dist_units);
    put_seq(w, seq);
    return w.bytes();
}

std::vector<uint8_t> encode_dmg(double t, int tick, int attacker, int victim,
                                float damage, int hitgroup, int weapon,
                                float dist_units, bool victim_visible,
                                uint32_t seq) {
    MsgpackWriter w;
    w.map(11);
    w.key("k"); w.str("dmg");
    w.key("t"); w.f64(t);
    w.key("tick"); w.i64(tick);
    w.key("a"); w.i64(attacker);
    w.key("v"); w.i64(victim);
    w.key("d"); w.f64(damage);
    w.key("hg"); w.i64(hitgroup);
    w.key("w"); w.i64(weapon);
    w.key("dist"); w.f64(dist_units);
    w.key("vis"); w.boolean(victim_visible);
    put_seq(w, seq);
    return w.bytes();
}

std::vector<uint8_t> encode_kill(double t, int tick, int attacker, int victim,
                                 int weapon, bool hs, float dist_units,
                                 bool smoke, bool wall, bool flash, bool noscope,
                                 float exp_ms, uint32_t seq) {
    MsgpackWriter w;
    w.map(14);
    w.key("k"); w.str("kill");
    w.key("t"); w.f64(t);
    w.key("tick"); w.i64(tick);
    w.key("a"); w.i64(attacker);
    w.key("v"); w.i64(victim);
    w.key("w"); w.i64(weapon);
    w.key("hs"); w.boolean(hs);
    w.key("dist"); w.f64(dist_units);
    w.key("smoke"); w.boolean(smoke);
    w.key("wall"); w.boolean(wall);
    w.key("flash"); w.boolean(flash);
    w.key("noscope"); w.boolean(noscope);
    w.key("exp_ms"); w.f64(exp_ms);
    put_seq(w, seq);
    return w.bytes();
}

std::vector<uint8_t> frame_packet(const std::vector<uint8_t>& payload,
                                  uint8_t flags) {
    std::vector<uint8_t> out;
    out.reserve(5 + payload.size());
    out.push_back(flags);
    const uint32_t n = uint32_t(payload.size());
    out.push_back(uint8_t(n >> 24));
    out.push_back(uint8_t(n >> 16));
    out.push_back(uint8_t(n >> 8));
    out.push_back(uint8_t(n));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

}  // namespace aegis
