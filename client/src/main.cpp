// Reference client for aegis/integrity/attest.py's challenge-response
// protocol -- see client/README.md for what this does and does not prove.
//
// Flow, matching Attestor exactly (aegis/integrity/attest.py):
//   1. POST /v1/attest/issue  -> {challenge_id, nonce_hex, deadline, probes}
//   2. Run exactly the probes the server asked for.
//   3. client_mac = HMAC-SHA256(device_key, nonce_bytes || measurement_bytes)
//   4. POST /v1/attest/verify -> {ok, reason, attested, untrusted, confirmed_tamper}

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <chrono>

#include "crypto.h"
#include "http_client.h"
#include "json_lite.h"
#include "probes.h"

using namespace aegis::client;

namespace {

std::string getenv_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : fallback;
}

// Lab-only default: SHA-256("aegis-dev-key"), byte-for-byte the same key
// Attestor() falls back to server-side when no operator key is configured
// (hashlib.sha256(b"aegis-dev-key").digest()). Real deployments must
// provision a real per-device secret through a secure channel and set
// AEGIS_DEVICE_KEY -- see client/README.md. This default exists so the
// protocol is testable end-to-end without one.
std::string default_device_key() {
    Digest32 d{};
    const std::string seed = "aegis-dev-key";
    sha256_bytes(reinterpret_cast<const uint8_t*>(seed.data()), seed.size(), d);
    return std::string(reinterpret_cast<const char*>(d.data()), d.size());
}

struct Args {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    std::string session_id = "lab-session";
    std::string player_id = "lab-player";
    bool loop = false;
    double interval_s = 30.0;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (arg == "--host") a.host = next();
        else if (arg == "--port") a.port = static_cast<uint16_t>(std::atoi(next().c_str()));
        else if (arg == "--session-id") a.session_id = next();
        else if (arg == "--player-id") a.player_id = next();
        else if (arg == "--loop") a.loop = true;
        else if (arg == "--interval") a.interval_s = std::atof(next().c_str());
    }
    return a;
}

// Returns true on a completed (not necessarily successful) round trip so
// --loop can keep going after a transient network failure.
bool run_once(const Args& a, const std::string& api_key, const std::string& device_key) {
    const std::string issue_body = json::object({
        "session_id", json::quote(a.session_id),
        "player_id", json::quote(a.player_id),
    });
    HttpResponse issue_resp = http_post_json(a.host, a.port, "/v1/attest/issue", issue_body, api_key);
    if (!issue_resp.ok || issue_resp.status != 200) {
        std::printf("[attest] issue failed: http_ok=%d status=%d err=%s body=%s\n",
                    issue_resp.ok, issue_resp.status, issue_resp.error.c_str(), issue_resp.body.c_str());
        return false;
    }

    const std::string challenge_id = json::extract_string(issue_resp.body, "challenge_id");
    const std::string nonce_hex = json::extract_string(issue_resp.body, "nonce_hex");
    const std::vector<std::string> probes = json::extract_string_array(issue_resp.body, "probes");
    if (challenge_id.empty() || nonce_hex.empty()) {
        std::printf("[attest] issue response missing fields: %s\n", issue_resp.body.c_str());
        return false;
    }
    std::printf("[attest] challenge %s probes=[", challenge_id.c_str());
    for (size_t i = 0; i < probes.size(); ++i) std::printf("%s%s", i ? "," : "", probes[i].c_str());
    std::printf("]\n");

    ProbeResults probe_results = run_probes(probes);
    const std::string measurement = build_measurement(probe_results, probes);
    const std::string nonce_bytes = from_hex(nonce_hex);

    Digest32 mac{};
    hmac_sha256(device_key, nonce_bytes + measurement, mac);

    std::string probe_ok_json = "{";
    bool first = true;
    for (const auto& [name, ok] : probe_ok_map(probe_results, probes)) {
        if (!first) probe_ok_json += ",";
        first = false;
        probe_ok_json += json::quote(name) + ":" + (ok ? "true" : "false");
    }
    probe_ok_json += "}";

    const std::string verify_body = json::object({
        "session_id", json::quote(a.session_id),
        "challenge_id", json::quote(challenge_id),
        "measurement_hex", json::quote(to_hex(
            reinterpret_cast<const uint8_t*>(measurement.data()), measurement.size())),
        "mac_hex", json::quote(to_hex(mac)),
        "probe_ok", probe_ok_json,
    });
    HttpResponse verify_resp = http_post_json(a.host, a.port, "/v1/attest/verify", verify_body, api_key);
    if (!verify_resp.ok) {
        std::printf("[attest] verify request failed: %s\n", verify_resp.error.c_str());
        return false;
    }

    const bool result_ok = json::extract_bool(verify_resp.body, "ok");
    const std::string reason = json::extract_string(verify_resp.body, "reason");
    const bool attested = json::extract_bool(verify_resp.body, "attested");
    const bool untrusted = json::extract_bool(verify_resp.body, "untrusted");
    const bool tamper = json::extract_bool(verify_resp.body, "confirmed_tamper");
    std::printf("[attest] status=%d ok=%d reason=%s attested=%d untrusted=%d confirmed_tamper=%d\n",
                verify_resp.status, result_ok, reason.c_str(), attested, untrusted, tamper);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    const std::string api_key = getenv_or("AEGIS_API_KEY", "");
    std::string device_key = getenv_or("AEGIS_DEVICE_KEY", "");
    if (device_key.empty()) {
        std::printf("[attest] WARNING: AEGIS_DEVICE_KEY not set; using the lab-only default. "
                    "This must be a real, securely-provisioned per-device secret before this "
                    "talks to anything but a local test server.\n");
        device_key = default_device_key();
    }

    do {
        run_once(a, api_key, device_key);
        if (a.loop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<long long>(a.interval_s * 1000)));
        }
    } while (a.loop);

    return 0;
}
