#include "serve/metrics.h"

#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::serve;

int main() {
    int failures     = 0;
    const auto check = [&](bool pass, const char* message) {
        if (!pass) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    EngineOptions options;
    options.max_concurrency = 2;
    options.speculative     = {.backend = SpeculativeBackend::Mtp,
                               .draft_tokens = 3,
                               .proposal_head = ProposalHead::Optimized};
    RuntimeStats baseline;
    baseline.committed_decode_tokens = 4;
    baseline.decode_rounds           = 1;
    Metrics metrics;
    metrics.configure("custom\"model\\name\nline", options, baseline);
    RuntimeStats running = baseline;
    running.committed_decode_tokens += 5;
    running.decode_rounds += 1;
    running.running_requests     = 1;
    running.device_main_kv_occupied_pages = 7;
    GenerationOutcome outcome;
    outcome.finish_reason               = FinishReason::OutputLimit;
    outcome.metrics.total_seconds       = 2.0;
    outcome.metrics.ttft_seconds        = 0.25;
    outcome.metrics.engine_timing.queue_wait_seconds = 0.1;
    outcome.metrics.speculative_backend             = SpeculativeBackend::Mtp;
    outcome.metrics.speculative_draft_window        = 3;
    outcome.metrics.speculative_rounds              = 2;
    outcome.metrics.speculative_draft_tokens        = 6;
    outcome.metrics.speculative_accepted_tokens     = 4;
    outcome.metrics.speculative_fallback_steps      = 1;
    metrics.done(outcome);
    const auto live = metrics.render(running, true);
    check(live.find("ninfer_engine_ready 1\n") != std::string::npos,
          "available engine must be observable");
    check(live.find("model_name=\"custom\\\"model\\\\name\\nline\"") != std::string::npos,
          "model label must escape quotes, backslashes and newlines");
    check(live.find("speculative_backend=\"mtp\"") != std::string::npos,
          "model info must name the configured speculative backend");
    check(live.find("ninfer_max_concurrency 2\n") != std::string::npos &&
              live.find("ninfer_spec_decode_draft_window 3\n") != std::string::npos,
          "startup limits must be exposed as gauges");
    check(live.find("ninfer_decode_tokens_total 5\n") != std::string::npos &&
              live.find("ninfer_decode_rounds_total 1\n") != std::string::npos,
          "live counters must exclude the startup warmup baseline");
    check(live.find("ninfer_requests_running 1\n") != std::string::npos &&
              live.find("ninfer_context_kv_pages_main_device 7\n") != std::string::npos,
          "runtime gauges must reflect the current snapshot");
    check(live.find("ninfer_spec_decode_draft_tokens_total 6\n") != std::string::npos &&
              live.find("ninfer_spec_decode_accepted_tokens_total 4\n") != std::string::npos,
          "speculative work must be aggregated from settled outcomes");
    check(live.find("ninfer_time_to_first_token_seconds_count 1\n") != std::string::npos &&
              live.find("ninfer_requests_total{outcome=\"completed\"} 1\n") != std::string::npos,
          "TTFT and completion must be recorded at settlement");
    check(live.find("_bucket{le=\"0.25\"} 1\n") != std::string::npos &&
              live.find("ninfer_time_to_first_token_seconds_sum 0.25\n") != std::string::npos,
          "TTFT must use inclusive histogram buckets");
    check(live.find("ninfer_request_queue_seconds_count 1\n") != std::string::npos,
          "queue wait must be observed per settled request");
    check(metrics.render(running, true) == live, "scrapes must not consume or reset counters");

    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([&] {
            for (int n = 0; n < 100; ++n) { metrics.done(outcome); }
        });
    }
    for (auto& writer : writers) { writer.join(); }
    outcome.finish_reason = FinishReason::Cancelled;
    metrics.done(outcome);
    metrics.failed(true);
    metrics.failed(false);
    metrics.rejected();
    metrics.response_failed();
    const auto final = metrics.render(running, false);
    check(final.find("ninfer_engine_ready 0\n") != std::string::npos,
          "unavailable engine must remain observable");
    check(final.find("ninfer_requests_total{outcome=\"completed\"} 401\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"cancelled\"} 2\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"failed\"} 1\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"rejected\"} 1\n") != std::string::npos,
          "concurrent settlements and terminal classifications must be preserved");
    check(final.find("ninfer_request_duration_seconds_bucket{le=\"+Inf\"} 402\n") !=
                  std::string::npos &&
              final.find("ninfer_request_duration_seconds_count 402\n") != std::string::npos,
          "histogram counts must include cancelled outcomes");
    check(final.find("ninfer_time_to_first_token_seconds_count 402\n") != std::string::npos,
          "every settled generation must contribute exactly one TTFT observation");
    check(final.find("ninfer_spec_decode_rounds_total 804\n") != std::string::npos &&
              final.find("ninfer_spec_decode_accepted_tokens_total 1608\n") != std::string::npos,
          "concurrent speculative aggregation must not lose outcomes");
    check(final.find("ninfer_response_failures_total 1\n") != std::string::npos,
          "response failures must be counted separately from request failures");
    return failures == 0 ? 0 : 1;
}
