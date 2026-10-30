#include "serve/metrics.h"

#include "product/speculative_options.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr std::array kBuckets{0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1,   0.25,  0.5,   1.0,
                              2.5,   5.0,   10.0,  20.0, 40.0,  80.0, 160.0, 320.0, 640.0, 1280.0};

std::string escape_label(std::string_view value) {
    std::string out;
    for (char c : value) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

} // namespace

void Metrics::Histogram::observe(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) { return; }
    ++bins[std::lower_bound(kBuckets.begin(), kBuckets.end(), seconds) - kBuckets.begin()];
    sum += seconds;
}

void Metrics::configure(std::string model, const EngineOptions& options, RuntimeStats baseline) {
    model_         = std::move(model);
    options_       = options;
    baseline_      = baseline;
    start_seconds_ =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void Metrics::done(const GenerationOutcome& outcome) {
    std::lock_guard lock(mutex_);
    if (outcome.finish_reason == FinishReason::Cancelled) {
        ++requests_.cancelled;
    } else {
        ++requests_.completed;
    }
    const auto& metrics = outcome.metrics;
    requests_.speculative_rounds += metrics.speculative_rounds;
    requests_.speculative_draft_tokens += metrics.speculative_draft_tokens;
    requests_.speculative_accepted += metrics.speculative_accepted_tokens;
    requests_.speculative_fallback_steps += metrics.speculative_fallback_steps;
    requests_.ttft.observe(metrics.ttft_seconds);
    requests_.duration.observe(metrics.total_seconds);
    requests_.queue.observe(metrics.engine_timing.queue_wait_seconds);
}

void Metrics::rejected() {
    std::lock_guard lock(mutex_);
    ++requests_.rejected;
}

void Metrics::failed(bool cancelled) {
    std::lock_guard lock(mutex_);
    if (cancelled) {
        ++requests_.cancelled;
    } else {
        ++requests_.failed;
    }
}

void Metrics::response_failed() {
    std::lock_guard lock(mutex_);
    ++requests_.response_failures;
}

std::string Metrics::render(const RuntimeStats& stats, bool ready) const {
    Requests requests;
    {
        std::lock_guard lock(mutex_);
        requests = requests_;
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    const auto header = [&](std::string_view name, std::string_view type, std::string_view help) {
        out << "# HELP ninfer_" << name << ' ' << help << '\n'
            << "# TYPE ninfer_" << name << ' ' << type << '\n';
    };
    const auto gauge = [&](std::string_view name, auto value, std::string_view help) {
        header(name, "gauge", help);
        out << "ninfer_" << name << ' ' << value << '\n';
    };
    const auto counter = [&](std::string_view name, auto current, auto baseline,
                             std::string_view help) {
        header(name, "counter", help);
        out << "ninfer_" << name << ' ' << current - baseline << '\n';
    };
    gauge("engine_ready", ready ? 1 : 0, "Whether Engine can accept work.");
    gauge("server_start_time_seconds", start_seconds_,
          "Unix time at service attachment after warmup.");
    header("model_info", "gauge", "Resident model and speculative backend.");
    out << "ninfer_model_info{model_name=\"" << escape_label(model_) << "\",speculative_backend=\""
        << product::speculative_backend_name(options_.speculative.backend) << "\"} 1\n";
    gauge("max_concurrency", options_.max_concurrency, "Maximum resident execution lanes.");
    gauge("max_context_tokens", options_.max_context, "Per-request logical context limit.");
    gauge("spec_decode_draft_window", options_.speculative.draft_tokens,
          "Configured speculative draft window.");
    {
#define GAUGE(field, name, help) gauge(name, stats.field, help)
        GAUGE(running_requests, "requests_running", "Resident requests, including prefill and decode.");
        GAUGE(waiting_requests, "requests_waiting", "Requests waiting for initial admission.");
        GAUGE(prefilling_requests, "requests_prefilling", "Resident requests in initial prefill.");
        GAUGE(decode_ready_requests, "requests_decode_ready", "Resident requests ready for decode.");
        GAUGE(materializing_requests, "requests_materializing",
              "Requests with a binding transaction in progress.");
        GAUGE(capture_pending_requests, "requests_capture_pending",
              "Resident requests waiting for a graph capture decision.");
        GAUGE(terminal_pending_requests, "requests_terminal_pending",
              "Resident requests with a settled outcome awaiting retirement.");
#undef GAUGE
    }
#define COUNTER(field, name, help) counter(name, stats.field, baseline_.field, help)
    COUNTER(computed_prefill_tokens, "prefill_tokens_total",
            "Initial input tokens actually computed by prefill.");
    COUNTER(reused_prompt_tokens, "prompt_tokens_cached_total",
            "Input tokens supplied by exact prefix reuse.");
    COUNTER(committed_decode_tokens, "decode_tokens_total",
            "Committed decode/control tokens, excluding the first token.");
    COUNTER(decode_rounds, "decode_rounds_total", "Decode batch executions.");
    COUNTER(decode_row_rounds, "decode_row_rounds_total", "Sum of decode batch sizes.");
    header("context_selections_total", "counter",
           "Initial binding reuse decisions by checkpoint kind.");
    {
        const std::pair<const char*, std::uint64_t> selections[] = {
            {"root", stats.root_selections},
            {"private_endpoint", stats.private_endpoint_selections},
            {"private_turn_closure", stats.private_turn_closure_selections},
            {"private_response_replay", stats.private_response_replay_selections},
            {"private_long_anchor", stats.private_long_anchor_selections},
            {"shared_stable_prefix", stats.shared_stable_prefix_selections}};
        for (const auto& [kind, value] : selections)
            out << "ninfer_context_selections_total{kind=\"" << kind << "\"} " << value << '\n';
    }
    COUNTER(historical_fork_hits, "context_historical_fork_hits_total",
            "Reuse decisions satisfied by recorded historical forks.");
    COUNTER(state_moves, "context_state_moves_total", "Logical State object relocations.");
    COUNTER(state_forks, "context_state_forks_total", "Logical State object forks.");
    COUNTER(state_restores, "context_state_restores_total", "Logical State object restores.");
    COUNTER(active_captures_completed, "active_captures_completed_total",
            "Completed graph capture windows.");
    COUNTER(active_captures_aborted, "active_captures_aborted_total", "Aborted graph captures.");
    gauge("context_state_slots_device", stats.device_state_occupied_slots,
          "Occupied Device State slots.");
    gauge("context_state_slots_host", stats.host_state_occupied_slots,
          "Occupied Host State slots.");
    gauge("context_kv_pages_main_device", stats.device_main_kv_occupied_pages,
          "Occupied Device Main KV pages.");
    gauge("context_kv_pages_backend_device", stats.device_backend_kv_occupied_pages,
          "Occupied Device Backend KV pages.");
    gauge("context_kv_bytes_host", stats.host_kv_occupied_bytes, "Occupied Host KV bytes.");
    gauge("context_shared_active_references", stats.shared_active_references,
          "Live shared-state references.");
    header("context_pressure_events_total", "counter",
           "Capacity-pressure reactions recorded by the Context coordinator.");
    {
        const std::pair<const char*, std::uint64_t> events[] = {
            {"spill_pages", stats.pressure_spill_pages},
            {"partial_tail_cow_pages", stats.partial_tail_cow_pages},
            {"private_owners_degraded", stats.pressure_private_owners_degraded},
            {"private_owners_evicted", stats.pressure_private_owners_evicted},
            {"shared_owners_degraded", stats.pressure_shared_owners_degraded},
            {"shared_owners_evicted", stats.pressure_shared_owners_evicted},
            {"checkpoints_dropped", stats.pressure_checkpoints_dropped},
            {"searches", stats.pressure_searches},
            {"search_budget_exhaustions", stats.pressure_search_budget_exhaustions},
            {"maximal_fallback_selections", stats.pressure_maximal_fallback_selections}};
        for (const auto& [kind, value] : events)
            out << "ninfer_context_pressure_events_total{event=\"" << kind << "\"} " << value
                << '\n';
    }
    header("context_transfer_bytes_total", "counter", "Completed context payload transfers.");
    header("context_transfer_pages_total", "counter",
           "Completed KV page-group transfers; State transfers report bytes only.");
    header("context_transfer_operation_seconds_total", "counter",
           "Accumulated context transfer operation time by resource and direction.");
    {
        constexpr std::array directions{"d2h", "h2d", "d2d"};
        const auto emit = [&](std::string_view resource, std::string_view direction,
                              std::uint64_t bytes, std::uint64_t pages, double seconds) {
            out << "ninfer_context_transfer_bytes_total{resource=\"" << resource
                << "\",direction=\"" << direction << "\"} " << bytes << '\n';
            if (resource != "state") {
                out << "ninfer_context_transfer_pages_total{resource=\"" << resource
                    << "\",direction=\"" << direction << "\"} " << pages << '\n';
            }
            out << "ninfer_context_transfer_operation_seconds_total{resource=\"" << resource
                << "\",direction=\"" << direction << "\"} " << seconds << '\n';
        };
        emit("state", directions[0], stats.state_d2h_bytes, 0, stats.state_d2h_seconds);
        emit("state", directions[1], stats.state_h2d_bytes, 0, stats.state_h2d_seconds);
        emit("state", directions[2], stats.state_d2d_bytes, 0, stats.state_d2d_seconds);
        emit("main_kv", directions[0], stats.main_kv_d2h_bytes, stats.main_kv_d2h_pages,
             stats.main_kv_d2h_seconds);
        emit("main_kv", directions[1], stats.main_kv_h2d_bytes, stats.main_kv_h2d_pages,
             stats.main_kv_h2d_seconds);
        emit("main_kv", directions[2], stats.main_kv_d2d_bytes, stats.main_kv_d2d_pages,
             stats.main_kv_d2d_seconds);
        emit("backend_kv", directions[0], stats.backend_kv_d2h_bytes, stats.backend_kv_d2h_pages,
             stats.backend_kv_d2h_seconds);
        emit("backend_kv", directions[1], stats.backend_kv_h2d_bytes, stats.backend_kv_h2d_pages,
             stats.backend_kv_h2d_seconds);
        emit("backend_kv", directions[2], stats.backend_kv_d2d_bytes, stats.backend_kv_d2d_pages,
             stats.backend_kv_d2d_seconds);
    }
    counter("context_transfer_seconds_total", stats.actual_context_transfer_seconds, 0.0,
            "Accumulated context transfer operation time.");
    header("host_work_seconds_total", "counter",
           "Coordinator call wall time, excluding scheduler yield or device wait.");
    {
        const std::pair<const char*, std::uint64_t RuntimeHostWorkStats::*> phases[] = {
            {"engine_boundary", &RuntimeHostWorkStats::engine_boundary_ns},
            {"program_submit", &RuntimeHostWorkStats::program_submit_ns},
            {"program_post", &RuntimeHostWorkStats::program_post_ns},
            {"engine_commit_output", &RuntimeHostWorkStats::engine_commit_output_ns},
            {"engine_maintenance", &RuntimeHostWorkStats::engine_maintenance_ns},
            {"decode_host", &RuntimeHostWorkStats::decode_host_ns},
            {"prefill_host", &RuntimeHostWorkStats::prefill_host_ns},
            {"control_host", &RuntimeHostWorkStats::control_host_ns}};
        for (const auto& [phase, field] : phases)
            out << "ninfer_host_work_seconds_total{phase=\"" << phase << "\"} "
                << static_cast<double>(stats.host_work.*field - baseline_.host_work.*field) * 1e-9
                << '\n';
    }
#undef COUNTER

    header("requests_total", "counter",
           "Generation attempts entering preparation, by terminal outcome.");
    for (const auto& [outcome, count] : std::array{
             std::pair{"completed", requests.completed}, std::pair{"cancelled", requests.cancelled},
             std::pair{"failed", requests.failed}, std::pair{"rejected", requests.rejected}}) {
        out << "ninfer_requests_total{outcome=\"" << outcome << "\"} " << count << '\n';
    }
    counter("response_failures_total", requests.response_failures, 0U,
            "Response rendering, storage or transport failures after generation settlement.");
    counter("spec_decode_rounds_total", requests.speculative_rounds, 0U,
            "Native speculative verification rounds settled by completed requests.");
    counter("spec_decode_draft_tokens_total", requests.speculative_draft_tokens, 0U,
            "Draft tokens evaluated by native verification.");
    counter("spec_decode_accepted_tokens_total", requests.speculative_accepted, 0U,
            "Draft tokens accepted by native verification.");
    counter("spec_decode_fallback_steps_total", requests.speculative_fallback_steps, 0U,
            "Speculative fallback steps without drafts.");
    const auto histogram = [&](std::string_view name, const Histogram& value,
                               std::string_view help) {
        header(name, "histogram", help);
        std::uint64_t count = 0;
        for (std::size_t i = 0; i < value.bins.size(); ++i) {
            count += value.bins[i];
            out << "ninfer_" << name << "_bucket{le=\"";
            if (i == kBuckets.size()) {
                out << "+Inf";
            } else {
                out << kBuckets[i];
            }
            out << "\"} " << count << '\n';
        }
        out << "ninfer_" << name << "_sum " << value.sum << '\n'
            << "ninfer_" << name << "_count " << count << '\n';
    };
    histogram("time_to_first_token_seconds", requests.ttft,
              "Preparation through first committed token; observed once at request settlement.");
    histogram("request_duration_seconds", requests.duration,
              "Generation request duration through settlement, including cancellation.");
    histogram("request_queue_seconds", requests.queue,
              "Initial Engine queue wait of settled generation requests.");
    return out.str();
}

} // namespace ninfer::serve
