#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace aegis {

// Non-blocking producer / dedicated consumer. Game threads only push; never
// wait on the socket. Drop-oldest if the queue backs up past max_queue.
class NetClient {
public:
    NetClient();
    ~NetClient();

    void configure(std::string host, int port);
    void start();
    void stop();

    // Enqueue a length-delimited packet. Returns false if dropped.
    bool push(std::vector<uint8_t> packet);

    uint64_t dropped() const { return dropped_.load(); }
    bool connected() const { return connected_.load(); }

private:
    void loop();
    bool ensure_connected();
    bool send_all(const uint8_t* data, size_t n);

    std::string host_;
    int port_ = 9100;
    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};
    std::atomic<uint64_t> dropped_{0};
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::vector<uint8_t>> queue_;
    size_t max_queue_ = 4096;
    std::thread thr_;
#ifdef _WIN32
    uintptr_t sock_ = ~uintptr_t(0);
#else
    int sock_ = -1;
#endif
};

}  // namespace aegis
