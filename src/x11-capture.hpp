#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <string>

class x11_capture_session {
public:
    x11_capture_session();
    ~x11_capture_session();
    x11_capture_session(const x11_capture_session &) = delete;
    x11_capture_session &operator=(const x11_capture_session &) = delete;
    bool capture(const std::string &encoded_selection, cv::Mat &bgr);
    uint64_t resolved_window_id() const { return resolved_window_id_; }
    bool used_direct_window_fallback() const { return used_direct_window_fallback_; }
    const std::string &last_error() const { return last_error_; }

private:
    struct impl;
    impl *state_ = nullptr;
    std::string last_error_;
    uint64_t resolved_window_id_ = 0;
    bool used_direct_window_fallback_ = false;
};