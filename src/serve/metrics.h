#pragma once

#include "serve/generation_service.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>

namespace ninfer::serve {

// Fixed-size request aggregates. Runtime counters remain owned and published by Engine.
class Metrics {
public:
    void configure(std::string model, const EngineOptions& options, RuntimeStats baseline);
    void done(const GenerationOutcome& outcome);
    void rejected();
    void failed(bool cancelled);
    void response_failed();
    [[nodiscard]] std::string render(const RuntimeStats& snapshot, bool ready) const;

private:
    struct Histogram {
        // Disjoint bins while recording; cumulative buckets are produced only when scraped.
        std::array<std::uint64_t, 21> bins{};
        double sum = 0.0;
        void observe(double seconds);
    };

    struct Requests {
        std::uint64_t completed         = 0;
        std::uint64_t cancelled         = 0;
        std::uint64_t failed            = 0;
        std::uint64_t rejected          = 0;
        std::uint64_t response_failures = 0;
        // Speculative work is settled per outcome; Engine RuntimeStats does not publish it.
        std::uint64_t speculative_rounds         = 0;
        std::uint64_t speculative_draft_tokens   = 0;
        std::uint64_t speculative_accepted       = 0;
        std::uint64_t speculative_fallback_steps = 0;
        Histogram ttft;
        Histogram duration;
        Histogram queue;
    };

    std::string model_;
    EngineOptions options_;
    RuntimeStats baseline_;
    double start_seconds_ = 0.0;
    mutable std::mutex mutex_;
    Requests requests_;
};

} // namespace ninfer::serve
