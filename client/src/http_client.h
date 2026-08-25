#pragma once

#include <cstdint>
#include <string>

namespace aegis::client {

struct HttpResponse {
    bool ok = false;          // request completed and got a response at all
    int status = 0;
    std::string body;
    std::string error;        // set when ok == false
};

// Minimal WinHTTP POST -- the service is a local/adjacent sidecar
// (aegis serve), not a public endpoint this client roams the internet to
// reach, so no retry/backoff/proxy handling beyond what WinHTTP does itself.
HttpResponse http_post_json(const std::string& host, uint16_t port, const std::string& path,
                             const std::string& json_body, const std::string& api_key,
                             bool use_tls = false);

}  // namespace aegis::client
