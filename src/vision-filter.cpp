#include <obs/obs-module.h>
#include <obs/obs.h>
#include <obs/graphics/graphics.h>
#include <obs/graphics/effect.h>
#include <obs/graphics/image-file.h>
#include <obs/graphics/vec2.h>
#include <obs/graphics/vec4.h>
#include "x11-capture.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/video/tracking.hpp>

#include <array>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef VISION_DEBUG_LOGS
#define VISION_DEBUG_LOGS 0
#endif
#if VISION_DEBUG_LOGS
#define VISION_LOG(...) blog(__VA_ARGS__)
#else
#define VISION_LOG(...) do { } while (0)
#endif

namespace {
constexpr size_t MAX_FACE_BOXES = 8;

struct face_box {
    float x;
    float y;
    float width;
    float height;
    float angle = 0.0f;
    float eye_y = 0.0f;
};

struct face_track {
    face_box box;
    face_box velocity = {0.0f, 0.0f, 0.0f, 0.0f};
    int missed = 0;
};

struct vision_filter_data {
    obs_source_t *source = nullptr;
    gs_effect_t *effect = nullptr;

    bool face_only = false;

    gs_eparam_t *threshold_param = nullptr;
    gs_eparam_t *strength_param = nullptr;
    gs_eparam_t *outline_color_param = nullptr;
    gs_eparam_t *fill_color_param = nullptr;
    gs_eparam_t *texel_size_param = nullptr;
    gs_eparam_t *color_levels_param = nullptr;
    gs_eparam_t *saturation_param = nullptr;
    gs_eparam_t *fill_strength_param = nullptr;
    gs_eparam_t *comic_strength_param = nullptr;

    gs_eparam_t *face_count_param = nullptr;

    gs_eparam_t *face_box_color_param = nullptr;
    gs_eparam_t *face_box_width_param = nullptr;
    gs_eparam_t *sunglasses_box_param = nullptr;
    gs_eparam_t *sunglasses_angle_param = nullptr;
    gs_eparam_t *sunglasses_image_param = nullptr;
    gs_eparam_t *pirate_hat_box_param = nullptr;
    gs_eparam_t *pirate_hat_angle_param = nullptr;
    gs_eparam_t *pirate_hat_image_param = nullptr;
    gs_eparam_t *clown_mask_box_param = nullptr;
    gs_eparam_t *clown_mask_angle_param = nullptr;
    gs_eparam_t *clown_mask_image_param = nullptr;
    std::array<gs_eparam_t *, MAX_FACE_BOXES> face_box_params{};

    double threshold = 0.20;
    double strength = 1.0;
    double color_levels = 4.0;
    double saturation = 1.15;
    double fill_strength = 0.0;
    double comic_strength = 1.0;
    uint32_t color = 0xFF000000;
    uint32_t fill_color = 0xFFFFFFFF;

    bool face_detection = true;
    bool show_face_boxes = true;
    int face_interval = 3;
    double face_scale = 0.5;
    double face_min_size = 0.02;
    double tracking_smoothing = 0.45;
    double tracking_iou = 0.10;
    int tracking_hold = 5;
    bool flip_analysis_vertical = true;
    uint32_t face_box_color = 0xFF00FF00;
    double face_box_width = 0.004;
    bool sunglasses_overlay = false;
    bool pirate_hat_overlay = false;
    bool clown_mask_overlay = false;
    gs_image_file_t sunglasses_image{};
    gs_image_file_t pirate_hat_image{};
    gs_image_file_t clown_mask_image{};
    bool sunglasses_image_ready = false;
    bool pirate_hat_image_ready = false;
    bool clown_mask_image_ready = false;
    bool sunglasses_candidate_valid = false;
    bool sunglasses_locked = false;
    face_box sunglasses_candidate{};
    face_box sunglasses_locked_box{};
    std::chrono::steady_clock::time_point sunglasses_candidate_since{};
    bool pirate_hat_candidate_valid = false;
    bool pirate_hat_locked = false;
    face_box pirate_hat_candidate{};
    face_box pirate_hat_locked_box{};
    std::chrono::steady_clock::time_point pirate_hat_candidate_since{};
    bool clown_mask_candidate_valid = false;
    bool clown_mask_locked = false;
    face_box clown_mask_candidate{};
    face_box clown_mask_locked_box{};
    std::chrono::steady_clock::time_point clown_mask_candidate_since{};


    std::mutex detection_mutex;
    std::condition_variable detection_cv;
    cv::Mat pending_bgr;
    bool pending_ready = false;
    std::vector<face_box> latest_faces;
    std::atomic_bool current_detection_present{false};
    std::atomic<uint64_t> last_detection_ns{0};

    std::atomic_bool stopping{false};
    std::thread detector_thread;
    std::thread x11_capture_thread;
    std::atomic<uint64_t> capture_window_id{0};
    std::mutex capture_selection_mutex;
    std::string capture_selection;
    std::atomic<int> crop_top{0};
    std::atomic<int> crop_left{0};
    std::atomic<int> crop_right{0};
    std::atomic<int> crop_height{0};
    uint64_t capture_attempts = 0;
    uint64_t capture_failures = 0;
    uint64_t capture_successes = 0;
    bool crop_logged = false;
    cv::Ptr<cv::FaceDetectorYN> face_detector;
    bool face_detector_ready = false;

    uint64_t frame_counter = 0;
    std::atomic<uint64_t> submitted_frames{0};
    std::atomic<uint64_t> processed_frames{0};
    std::atomic<uint64_t> detector_runs{0};
    std::atomic<uint64_t> detection_hits{0};
    std::atomic<uint64_t> accepted_detections{0};
    bool first_frame_logged = false;

    bool source_chain_logged = false;
    bool image_param_logged = false;
};

static const char *contour_filter_name(void *)
{
    return obs_module_text("VisionContourFilter");
}

static const char *face_filter_name(void *)
{
    return obs_module_text("VisionFaceFilter");
}

static void contour_defaults(obs_data_t *settings)
{
    obs_data_set_double(settings, "threshold", 0.20);
    obs_data_set_double(settings, "strength", 1.0);
    obs_data_set_double(settings, "color_levels", 4.0);
    obs_data_set_double(settings, "saturation", 1.15);
    obs_data_set_double(settings, "fill_strength", 0.0);
    obs_data_set_double(settings, "comic_strength", 1.0);
    obs_data_set_int(settings, "color", 0xFF000000);
    obs_data_set_int(settings, "fill_color", 0xFFFFFFFF);
}

static void face_defaults(obs_data_t *settings)
{
    obs_data_set_bool(settings, "face_detection", true);
    obs_data_set_int(settings, "face_interval", 3);
    obs_data_set_double(settings, "face_scale", 0.5);
    obs_data_set_double(settings, "face_min_size", 0.02);
    obs_data_set_double(settings, "tracking_smoothing", 0.45);
    obs_data_set_double(settings, "tracking_iou", 0.10);
    obs_data_set_int(settings, "tracking_hold", 5);
    obs_data_set_bool(settings, "flip_analysis_vertical", false);
    obs_data_set_double(settings, "face_box_width", 0.004);
    obs_data_set_int(settings, "face_box_color", 0xFF00FF00);
    obs_data_set_bool(settings, "show_face_boxes", true);
    obs_data_set_bool(settings, "sunglasses_overlay", false);
    obs_data_set_bool(settings, "pirate_hat_overlay", false);
    obs_data_set_bool(settings, "clown_mask_overlay", false);

}

static obs_properties_t *contour_properties(void *)
{
    obs_properties_t *props = obs_properties_create();
    obs_properties_add_float_slider(props, "threshold", obs_module_text("Threshold"), 0.01, 1.0, 0.01);
    obs_properties_add_float_slider(props, "strength", obs_module_text("Strength"), 0.0, 2.0, 0.01);
    obs_properties_add_float_slider(props, "color_levels", obs_module_text("ColorLevels"), 2.0, 16.0, 1.0);
    obs_properties_add_float_slider(props, "saturation", obs_module_text("Saturation"), 0.0, 2.0, 0.01);
    obs_properties_add_float_slider(props, "comic_strength", obs_module_text("ComicStrength"), 0.0, 1.0, 0.01);
    obs_properties_add_float_slider(props, "fill_strength", obs_module_text("FillStrength"), 0.0, 1.0, 0.01);
    obs_properties_add_color(props, "color", obs_module_text("OutlineColor"));
    obs_properties_add_color(props, "fill_color", obs_module_text("FillColor"));
    return props;
}

static obs_properties_t *face_properties(void *)
{
    obs_properties_t *props = obs_properties_create();
    obs_properties_add_bool(props, "face_detection", obs_module_text("FaceDetection"));
    obs_properties_add_bool(props, "show_face_boxes", obs_module_text("ShowFaceBoxes"));
    obs_properties_add_int_slider(props, "face_interval", obs_module_text("FaceInterval"), 1, 10, 1);
    obs_properties_add_float_slider(props, "face_scale", obs_module_text("FaceScale"), 0.20, 1.0, 0.05);
    obs_properties_add_float_slider(props, "face_min_size", obs_module_text("FaceMinSize"), 0.02, 0.30, 0.01);
    obs_properties_add_float_slider(props, "tracking_smoothing", obs_module_text("TrackingSmoothing"), 0.05, 1.0, 0.05);
    obs_properties_add_float_slider(props, "tracking_iou", obs_module_text("TrackingIou"), 0.05, 0.80, 0.05);
    obs_properties_add_int_slider(props, "tracking_hold", obs_module_text("TrackingHold"), 0, 10, 1);
    obs_properties_add_bool(props, "flip_analysis_vertical", obs_module_text("FlipAnalysisVertical"));
    obs_properties_add_float_slider(props, "face_box_width", obs_module_text("FaceBoxWidth"), 0.001, 0.02, 0.001);
    obs_properties_add_color(props, "face_box_color", obs_module_text("FaceBoxColor"));
    obs_properties_add_bool(props, "sunglasses_overlay", obs_module_text("SunglassesOverlay"));
    obs_properties_add_bool(props, "pirate_hat_overlay", obs_module_text("PirateHatOverlay"));
    obs_properties_add_bool(props, "clown_mask_overlay", obs_module_text("ClownMaskOverlay"));

    return props;
}

static void contour_update(void *data, obs_data_t *settings)
{
    auto *filter = static_cast<vision_filter_data *>(data);
    filter->threshold = obs_data_get_double(settings, "threshold");
    filter->strength = obs_data_get_double(settings, "strength");
    filter->color_levels = obs_data_get_double(settings, "color_levels");
    filter->saturation = obs_data_get_double(settings, "saturation");
    filter->fill_strength = obs_data_get_double(settings, "fill_strength");
    filter->comic_strength = obs_data_get_double(settings, "comic_strength");
    filter->color = static_cast<uint32_t>(obs_data_get_int(settings, "color"));
    filter->fill_color = static_cast<uint32_t>(obs_data_get_int(settings, "fill_color"));
}

static void face_update(void *data, obs_data_t *settings)
{
    auto *filter = static_cast<vision_filter_data *>(data);
    filter->face_detection = obs_data_get_bool(settings, "face_detection");
    filter->show_face_boxes = obs_data_get_bool(settings, "show_face_boxes");
    filter->face_interval = static_cast<int>(obs_data_get_int(settings, "face_interval"));
    filter->face_scale = obs_data_get_double(settings, "face_scale");
    filter->face_min_size = obs_data_get_double(settings, "face_min_size");
    filter->tracking_smoothing = obs_data_get_double(settings, "tracking_smoothing");
    filter->tracking_iou = std::max(0.05, obs_data_get_double(settings, "tracking_iou"));
    filter->tracking_hold = std::max(1, static_cast<int>(obs_data_get_int(settings, "tracking_hold")));
    filter->flip_analysis_vertical = false;
    filter->face_box_width = obs_data_get_double(settings, "face_box_width");
    filter->face_box_color = static_cast<uint32_t>(obs_data_get_int(settings, "face_box_color"));
    filter->sunglasses_overlay = obs_data_get_bool(settings, "sunglasses_overlay");
    filter->pirate_hat_overlay = obs_data_get_bool(settings, "pirate_hat_overlay");
    filter->clown_mask_overlay = obs_data_get_bool(settings, "clown_mask_overlay");

}

static float box_iou(const face_box &a, const face_box &b)
{
    const float left = std::max(a.x, b.x);
    const float top = std::max(a.y, b.y);
    const float right = std::min(a.x + a.width, b.x + b.width);
    const float bottom = std::min(a.y + a.height, b.y + b.height);
    const float intersection = std::max(0.0f, right - left) *
                               std::max(0.0f, bottom - top);
    const float area_a = a.width * a.height;
    const float area_b = b.width * b.height;
    const float union_area = area_a + area_b - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static std::vector<face_box> update_tracks(std::vector<face_track> &tracks,
                                           const std::vector<face_box> &detections,
                                           float smoothing, float minimum_iou,
                                           int max_missed)
{
    std::vector<bool> used(detections.size(), false);
    const float alpha = std::clamp(smoothing, 0.05f, 1.0f);

    for (face_track &track : tracks) {
        int best_index = -1;
        float best_iou = minimum_iou;
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!used[i]) {
                const float overlap = box_iou(track.box, detections[i]);
                if (overlap > best_iou) {
                    best_iou = overlap;
                    best_index = static_cast<int>(i);
                }
            }
        }

        if (best_index >= 0) {
            const face_box &target = detections[best_index];
            track.velocity = {
                (target.x - track.box.x) * 0.5f,
                (target.y - track.box.y) * 0.5f,
                (target.width - track.box.width) * 0.5f,
                (target.height - track.box.height) * 0.5f,
            };
            track.box = {
                track.box.x + (target.x - track.box.x) * alpha,
                track.box.y + (target.y - track.box.y) * alpha,
                track.box.width + (target.width - track.box.width) * alpha,
                track.box.height + (target.height - track.box.height) * alpha,
                track.box.angle + (target.angle - track.box.angle) * alpha,
                track.box.eye_y + (target.eye_y - track.box.eye_y) * alpha,
            };
            track.missed = 0;
            used[best_index] = true;
        } else {
            track.box.x += track.velocity.x;
            track.box.y += track.velocity.y;
            track.box.width += track.velocity.width;
            track.box.height += track.velocity.height;
            track.velocity.x *= 0.8f;
            track.velocity.y *= 0.8f;
            track.velocity.width *= 0.8f;
            track.velocity.height *= 0.8f;
            ++track.missed;
        }
    }

    for (size_t i = 0; i < detections.size(); ++i) {
        if (!used[i])
            tracks.push_back({detections[i], {}, 0});
    }

    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                [max_missed](const face_track &track) {
                                    return track.missed > max_missed;
                                }),
                 tracks.end());

    std::vector<face_box> result;
    result.reserve(std::min(tracks.size(), MAX_FACE_BOXES));
    for (const face_track &track : tracks) {
        if (result.size() == MAX_FACE_BOXES)
            break;
        result.push_back(track.box);
    }
    return result;
}

static void detector_loop(vision_filter_data *filter)
{
    std::vector<face_track> tracks;
    cv::Mat previous_gray;

    for (;;) {
        cv::Mat bgr;
        {
            std::unique_lock<std::mutex> lock(filter->detection_mutex);
            filter->detection_cv.wait(lock, [filter] {
                return filter->stopping.load() || filter->pending_ready;
            });
            if (filter->stopping.load())
                return;
            bgr = std::move(filter->pending_bgr);
            filter->pending_ready = false;
        }

        if (bgr.empty())
            continue;
        ++filter->processed_frames;

        cv::Mat current_gray;
        cv::cvtColor(bgr, current_gray, cv::COLOR_BGR2GRAY);

        // Track existing boxes between detector passes with optical flow.
        if (!previous_gray.empty() && !tracks.empty() &&
            previous_gray.size() == current_gray.size()) {
            std::vector<cv::Point2f> previous_points;
            std::vector<size_t> point_track_indices;
            previous_points.reserve(tracks.size() * 5);
            point_track_indices.reserve(tracks.size() * 5);
            for (size_t track_index = 0; track_index < tracks.size(); ++track_index) {
                const face_track &track = tracks[track_index];
                const float left = (track.box.x + track.box.width * 0.25f) * previous_gray.cols;
                const float right = (track.box.x + track.box.width * 0.75f) * previous_gray.cols;
                const float top = (track.box.y + track.box.height * 0.25f) * previous_gray.rows;
                const float bottom = (track.box.y + track.box.height * 0.75f) * previous_gray.rows;
                const float center_x = (left + right) * 0.5f;
                const float center_y = (top + bottom) * 0.5f;
                const cv::Point2f points[] = {
                    {left, top}, {right, top}, {center_x, center_y},
                    {left, bottom}, {right, bottom},
                };
                for (const cv::Point2f &point : points) {
                    previous_points.push_back(point);
                    point_track_indices.push_back(track_index);
                }
            }

            std::vector<cv::Point2f> current_points;
            std::vector<uint8_t> status;
            std::vector<float> errors;
            try {
                cv::calcOpticalFlowPyrLK(previous_gray, current_gray,
                                         previous_points, current_points,
                                         status, errors);
            } catch (const cv::Exception &error) {
                VISION_LOG(LOG_WARNING, "Face tracking optical-flow reset: %s", error.what());
                current_points.clear();
                status.clear();
            }

            std::vector<std::vector<float>> dx_by_track(tracks.size());
            std::vector<std::vector<float>> dy_by_track(tracks.size());
            for (size_t i = 0; i < current_points.size() && i < status.size(); ++i) {
                if (!status[i])
                    continue;
                const size_t track_index = point_track_indices[i];
                const float dx = (current_points[i].x / current_gray.cols) -
                                 (previous_points[i].x / previous_gray.cols);
                const float dy = (current_points[i].y / current_gray.rows) -
                                 (previous_points[i].y / previous_gray.rows);
                dx_by_track[track_index].push_back(dx);
                dy_by_track[track_index].push_back(dy);
            }

            for (size_t track_index = 0; track_index < tracks.size(); ++track_index) {
                if (dx_by_track[track_index].empty())
                    continue;
                auto &dx_values = dx_by_track[track_index];
                auto &dy_values = dy_by_track[track_index];
                std::sort(dx_values.begin(), dx_values.end());
                std::sort(dy_values.begin(), dy_values.end());
                tracks[track_index].box.x += dx_values[dx_values.size() / 2];
                tracks[track_index].box.y += dy_values[dy_values.size() / 2];
            }
        }

        // face_interval controls GPU sampling in face_render(); every submitted
        // sample is a detector/tracker step here.
        const bool run_detection = filter->face_detector_ready;

        std::vector<face_box> normalized;
        if (run_detection && filter->face_detector_ready) {
            ++filter->detector_runs;
            cv::Mat detected;
            filter->face_detector->setInputSize(bgr.size());
            filter->face_detector->detect(bgr, detected);
            if (!detected.empty()) {
                ++filter->detection_hits;
                normalized.reserve(std::min(static_cast<size_t>(detected.rows), MAX_FACE_BOXES));
                for (int row = 0; row < detected.rows && normalized.size() < MAX_FACE_BOXES; ++row) {
                    const float x = detected.at<float>(row, 0);
                    const float y = detected.at<float>(row, 1);
                    const float width = detected.at<float>(row, 2);
                    const float height = detected.at<float>(row, 3);
                    const float score = detected.at<float>(row, 14);
                    if (score < 0.35f)
                        continue;
                    const float nx = std::clamp(x / bgr.cols, 0.0f, 1.0f);
                    const float ny = std::clamp(y / bgr.rows, 0.0f, 1.0f);
                    const float nw = std::clamp(width / bgr.cols, 0.0f, 1.0f - nx);
                    const float nh = std::clamp(height / bgr.rows, 0.0f, 1.0f - ny);
                    if (nw <= 0.0f || nh <= 0.0f)
                        continue;
                    const float right_eye_x = detected.at<float>(row, 4);
                    const float right_eye_y = detected.at<float>(row, 5);
                    const float left_eye_x = detected.at<float>(row, 6);
                    const float left_eye_y = detected.at<float>(row, 7);
                    const float angle = std::atan2(left_eye_y - right_eye_y,
                                                   left_eye_x - right_eye_x);
                    const float eye_y = std::clamp(
                        ((right_eye_y + left_eye_y) * 0.5f) / bgr.rows, 0.0f, 1.0f);
                    normalized.push_back({nx, ny, nw, nh, angle, eye_y});
                    ++filter->accepted_detections;
                }
            }
        }

        std::vector<face_box> tracked;
        if (run_detection) {
            tracked = update_tracks(
                tracks, normalized,
                static_cast<float>(filter->tracking_smoothing),
                static_cast<float>(filter->tracking_iou),
                filter->tracking_hold);
            const bool present = !normalized.empty();
            filter->current_detection_present.store(present);
            filter->last_detection_ns.store(
                present ? static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count()) : 0);
        } else {
            filter->current_detection_present.store(false);
            filter->last_detection_ns.store(0);
            tracked.reserve(std::min(tracks.size(), MAX_FACE_BOXES));
            for (const face_track &track : tracks) {
                if (tracked.size() == MAX_FACE_BOXES)
                    break;
                tracked.push_back(track.box);
            }
        }

        previous_gray = std::move(current_gray);
        std::lock_guard<std::mutex> lock(filter->detection_mutex);
        filter->latest_faces = std::move(tracked);
    }
}

static void set_color(gs_eparam_t *param, uint32_t packed)
{
    const float a = static_cast<float>((packed >> 24) & 0xFF) / 255.0f;
    const float r = static_cast<float>((packed >> 16) & 0xFF) / 255.0f;
    const float g = static_cast<float>((packed >> 8) & 0xFF) / 255.0f;
    const float b = static_cast<float>(packed & 0xFF) / 255.0f;
    struct vec4 color = {r, g, b, a};
    gs_effect_set_vec4(param, &color);
}

static bool detection_is_alive(const vision_filter_data *filter)
{
    if (!filter->current_detection_present.load())
        return false;
    const uint64_t last = filter->last_detection_ns.load();
    if (last == 0)
        return false;
    const uint64_t now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    return now >= last && now - last <= 300000000ULL;
}

static bool update_sunglasses_target(vision_filter_data *filter,
                                     const std::vector<face_box> &faces,
                                     face_box &target)
{
    if (!filter->sunglasses_overlay || !detection_is_alive(filter) || faces.empty()) {
        filter->sunglasses_candidate_valid = false;
        filter->sunglasses_locked = false;
        return false;
    }

    const auto biggest = *std::max_element(
        faces.begin(), faces.end(), [](const face_box &a, const face_box &b) {
            return a.width * a.height < b.width * b.height;
        });
    const auto now = std::chrono::steady_clock::now();
    constexpr float target_iou = 0.25f;
    constexpr auto lock_delay = std::chrono::milliseconds(600);

    if (!filter->sunglasses_locked) {
        if (!filter->sunglasses_candidate_valid ||
            box_iou(filter->sunglasses_candidate, biggest) < target_iou) {
            filter->sunglasses_candidate = biggest;
            filter->sunglasses_candidate_since = now;
            filter->sunglasses_candidate_valid = true;
        } else {
            filter->sunglasses_candidate = biggest;
        }

        if (now - filter->sunglasses_candidate_since >= lock_delay) {
            filter->sunglasses_locked = true;
            filter->sunglasses_locked_box = biggest;
            VISION_LOG(LOG_INFO, "Face sunglasses target locked after 600 ms");
        }
    } else if (box_iou(filter->sunglasses_locked_box, biggest) >= target_iou) {
        filter->sunglasses_locked_box = biggest;
        filter->sunglasses_candidate_valid = false;
    } else {
        if (!filter->sunglasses_candidate_valid ||
            box_iou(filter->sunglasses_candidate, biggest) < target_iou) {
            filter->sunglasses_candidate = biggest;
            filter->sunglasses_candidate_since = now;
            filter->sunglasses_candidate_valid = true;
        } else {
            filter->sunglasses_candidate = biggest;
        }

        if (now - filter->sunglasses_candidate_since >= lock_delay) {
            filter->sunglasses_locked_box = biggest;
            filter->sunglasses_candidate_valid = false;
            VISION_LOG(LOG_INFO, "Face sunglasses target switched after 600 ms");
        }
    }

    if (!filter->sunglasses_locked)
        return false;
    target = filter->sunglasses_locked_box;
    return true;
}

static bool update_pirate_hat_target(vision_filter_data *filter,
                                     const std::vector<face_box> &faces,
                                     face_box &target)
{
    if (!filter->pirate_hat_overlay || !detection_is_alive(filter) || faces.size() < 2) {
        filter->pirate_hat_candidate_valid = false;
        filter->pirate_hat_locked = false;
        return false;
    }

    size_t largest = 0;
    size_t second_largest = 1;
    if (faces[second_largest].width * faces[second_largest].height >
        faces[largest].width * faces[largest].height)
        std::swap(largest, second_largest);
    for (size_t i = 2; i < faces.size(); ++i) {
        const float area = faces[i].width * faces[i].height;
        const float second_area = faces[second_largest].width * faces[second_largest].height;
        if (area > faces[largest].width * faces[largest].height) {
            second_largest = largest;
            largest = i;
        } else if (area > second_area) {
            second_largest = i;
        }
    }
    const face_box second_face = faces[second_largest];
    const auto now = std::chrono::steady_clock::now();
    constexpr float target_iou = 0.25f;
    constexpr auto lock_delay = std::chrono::milliseconds(600);

    if (!filter->pirate_hat_locked) {
        if (!filter->pirate_hat_candidate_valid ||
            box_iou(filter->pirate_hat_candidate, second_face) < target_iou) {
            filter->pirate_hat_candidate = second_face;
            filter->pirate_hat_candidate_since = now;
            filter->pirate_hat_candidate_valid = true;
        } else {
            filter->pirate_hat_candidate = second_face;
        }
        if (now - filter->pirate_hat_candidate_since >= lock_delay) {
            filter->pirate_hat_locked = true;
            filter->pirate_hat_locked_box = second_face;
            VISION_LOG(LOG_INFO, "Face pirate hat target locked after 600 ms");
        }
    } else if (box_iou(filter->pirate_hat_locked_box, second_face) >= target_iou) {
        filter->pirate_hat_locked_box = second_face;
        filter->pirate_hat_candidate_valid = false;
    } else {
        if (!filter->pirate_hat_candidate_valid ||
            box_iou(filter->pirate_hat_candidate, second_face) < target_iou) {
            filter->pirate_hat_candidate = second_face;
            filter->pirate_hat_candidate_since = now;
            filter->pirate_hat_candidate_valid = true;
        } else {
            filter->pirate_hat_candidate = second_face;
        }
        if (now - filter->pirate_hat_candidate_since >= lock_delay) {
            filter->pirate_hat_locked_box = second_face;
            filter->pirate_hat_candidate_valid = false;
            VISION_LOG(LOG_INFO, "Face pirate hat target switched after 600 ms");
        }
    }

    if (!filter->pirate_hat_locked)
        return false;
    target = filter->pirate_hat_locked_box;
    return true;
}

static bool update_clown_mask_target(vision_filter_data *filter,
                                     const std::vector<face_box> &faces,
                                     face_box &target)
{
    if (!filter->clown_mask_overlay || !detection_is_alive(filter) || faces.size() < 3) {
        filter->clown_mask_candidate_valid = false;
        filter->clown_mask_locked = false;
        return false;
    }

    std::vector<face_box> sorted_faces = faces;
    std::sort(sorted_faces.begin(), sorted_faces.end(), [](const face_box &a, const face_box &b) {
        return a.width * a.height > b.width * b.height;
    });
    const face_box third_face = sorted_faces[2];
    const auto now = std::chrono::steady_clock::now();
    constexpr float target_iou = 0.25f;
    constexpr auto lock_delay = std::chrono::milliseconds(600);

    if (!filter->clown_mask_locked) {
        if (!filter->clown_mask_candidate_valid ||
            box_iou(filter->clown_mask_candidate, third_face) < target_iou) {
            filter->clown_mask_candidate = third_face;
            filter->clown_mask_candidate_since = now;
            filter->clown_mask_candidate_valid = true;
        } else {
            filter->clown_mask_candidate = third_face;
        }
        if (now - filter->clown_mask_candidate_since >= lock_delay) {
            filter->clown_mask_locked = true;
            filter->clown_mask_locked_box = third_face;
            VISION_LOG(LOG_INFO, "Face clown mask target locked after 600 ms");
        }
    } else if (box_iou(filter->clown_mask_locked_box, third_face) >= target_iou) {
        filter->clown_mask_locked_box = third_face;
        filter->clown_mask_candidate_valid = false;
    } else {
        if (!filter->clown_mask_candidate_valid ||
            box_iou(filter->clown_mask_candidate, third_face) < target_iou) {
            filter->clown_mask_candidate = third_face;
            filter->clown_mask_candidate_since = now;
            filter->clown_mask_candidate_valid = true;
        } else {
            filter->clown_mask_candidate = third_face;
        }
        if (now - filter->clown_mask_candidate_since >= lock_delay) {
            filter->clown_mask_locked_box = third_face;
            filter->clown_mask_candidate_valid = false;
            VISION_LOG(LOG_INFO, "Face clown mask target switched after 600 ms");
        }
    }

    if (!filter->clown_mask_locked)
        return false;
    target = filter->clown_mask_locked_box;
    return true;
}

static void render_faces(vision_filter_data *filter, uint32_t source_width,
                         uint32_t source_height)
{
    std::vector<face_box> faces;
    {
        std::lock_guard<std::mutex> lock(filter->detection_mutex);
        faces = filter->latest_faces;
    }

    gs_effect_set_int(filter->face_count_param,
                      filter->face_detection && filter->show_face_boxes
                          ? static_cast<int>(faces.size()) : 0);
    struct vec4 sunglasses_box = {0.0f, 0.0f, 0.0f, 0.0f};
    face_box target{};
    if (filter->face_detection && filter->sunglasses_image_ready &&
        update_sunglasses_target(filter, faces, target)) {
        const float image_aspect = filter->sunglasses_image.cy > 0
            ? static_cast<float>(filter->sunglasses_image.cx) /
              static_cast<float>(filter->sunglasses_image.cy)
            : 1.0f;
        const float overlay_width = target.width;
        const float overlay_height = overlay_width *
            (static_cast<float>(source_width) / static_cast<float>(source_height)) /
            image_aspect;
        const float eye_y = filter->flip_analysis_vertical
            ? 1.0f - target.eye_y
            : target.eye_y;
        sunglasses_box = {
            target.x,
            eye_y - overlay_height * 0.5f,
            overlay_width,
            overlay_height,
        };
    }
    gs_effect_set_vec4(filter->sunglasses_box_param, &sunglasses_box);
    gs_effect_set_float(filter->sunglasses_angle_param,
                        target.angle);
    gs_effect_set_texture(filter->sunglasses_image_param,
                          filter->sunglasses_image.texture);

    struct vec4 pirate_hat_box = {0.0f, 0.0f, 0.0f, 0.0f};
    float pirate_hat_angle = 0.0f;
    face_box hat_target{};
    if (filter->face_detection && filter->pirate_hat_image_ready &&
        update_pirate_hat_target(filter, faces, hat_target)) {
        const face_box &target = hat_target;
        const float image_aspect = filter->pirate_hat_image.cy > 0
            ? static_cast<float>(filter->pirate_hat_image.cx) /
              static_cast<float>(filter->pirate_hat_image.cy)
            : 1.0f;
        const float hat_width = target.width * 2.00f;
        const float hat_height = hat_width *
            (static_cast<float>(source_width) / static_cast<float>(source_height)) /
            image_aspect;
        const float target_y = filter->flip_analysis_vertical
            ? 1.0f - (target.y + target.height)
            : target.y;
        pirate_hat_box = {
            target.x - (hat_width - target.width) * 0.5f,
            target_y - hat_height * 0.58f,
            hat_width,
            hat_height,
        };
        pirate_hat_angle = target.angle;
    }
    gs_effect_set_vec4(filter->pirate_hat_box_param, &pirate_hat_box);
    gs_effect_set_float(filter->pirate_hat_angle_param, pirate_hat_angle);
    gs_effect_set_texture(filter->pirate_hat_image_param,
                          filter->pirate_hat_image.texture);

    struct vec4 clown_mask_box = {0.0f, 0.0f, 0.0f, 0.0f};
    float clown_mask_angle = 0.0f;
    face_box mask_target{};
    if (filter->face_detection && filter->clown_mask_image_ready &&
        update_clown_mask_target(filter, faces, mask_target)) {
        const float image_aspect = filter->clown_mask_image.cy > 0
            ? static_cast<float>(filter->clown_mask_image.cx) /
              static_cast<float>(filter->clown_mask_image.cy)
            : 1.0f;
        const float mask_width = mask_target.width * 1.35f;
        const float mask_height = mask_width *
            (static_cast<float>(source_width) / static_cast<float>(source_height)) /
            image_aspect;
        const float target_y = filter->flip_analysis_vertical
            ? 1.0f - (mask_target.y + mask_target.height)
            : mask_target.y;
        const float center_x = mask_target.x + mask_target.width * 0.5f;
        const float center_y = target_y + mask_target.height * 0.5f;
        clown_mask_box = {
            center_x - mask_width * 0.5f,
            center_y - mask_height * 0.5f,
            mask_width,
            mask_height,
        };
        clown_mask_angle = mask_target.angle;
    }
    gs_effect_set_vec4(filter->clown_mask_box_param, &clown_mask_box);
    gs_effect_set_float(filter->clown_mask_angle_param, clown_mask_angle);
    gs_effect_set_texture(filter->clown_mask_image_param,
                          filter->clown_mask_image.texture);
    for (size_t i = 0; i < MAX_FACE_BOXES; ++i) {
        struct vec4 box = {0.0f, 0.0f, 0.0f, 0.0f};
        if (i < faces.size()) {
            const face_box &face = faces[i];
            box = {face.x, filter->flip_analysis_vertical
                                  ? 1.0f - (face.y + face.height)
                                  : face.y,
                   face.width, face.height};
        }
        gs_effect_set_vec4(filter->face_box_params[i], &box);

    }
    set_color(filter->face_box_color_param, filter->face_box_color);
    gs_effect_set_float(filter->face_box_width_param,
                        static_cast<float>(filter->face_box_width));
}

static void contour_render(void *data, gs_effect_t *)
{
    auto *filter = static_cast<vision_filter_data *>(data);
    obs_source_t *parent = obs_filter_get_parent(filter->source);
    if (!parent || !obs_source_process_filter_begin_with_color_space(
                         filter->source, GS_RGBA, GS_CS_SRGB,
                         OBS_ALLOW_DIRECT_RENDERING))
        return;

    const uint32_t width = obs_source_get_base_width(parent);
    const uint32_t height = obs_source_get_base_height(parent);
    if (width == 0 || height == 0 || !filter->effect) {
        obs_source_process_filter_end(filter->source, filter->effect, 0, 0);
        return;
    }

    struct vec2 texel_size = {1.0f / width, 1.0f / height};
    gs_effect_set_float(filter->threshold_param, static_cast<float>(filter->threshold));
    gs_effect_set_float(filter->strength_param, static_cast<float>(filter->strength));
    gs_effect_set_float(filter->color_levels_param, static_cast<float>(filter->color_levels));
    gs_effect_set_float(filter->saturation_param, static_cast<float>(filter->saturation));
    gs_effect_set_float(filter->fill_strength_param, static_cast<float>(filter->fill_strength));
    gs_effect_set_float(filter->comic_strength_param, static_cast<float>(filter->comic_strength));
    gs_effect_set_vec2(filter->texel_size_param, &texel_size);
    set_color(filter->outline_color_param, filter->color);
    set_color(filter->fill_color_param, filter->fill_color);
    obs_source_process_filter_end(filter->source, filter->effect, 0, 0);
}

static void submit_detection_bgr(vision_filter_data *filter, cv::Mat bgr);

static void refresh_capture_window(vision_filter_data *filter, obs_source_t *parent)
{
    obs_data_t *settings = obs_source_get_settings(parent);
    if (!settings)
        return;
    const char *value = obs_data_get_string(settings, "capture_window");
    std::string raw = value ? value : "";
    for (char &ch : raw) {
        if (ch == '\r' || ch == '\n')
            ch = '|';
    }
    uint64_t window_id = 0;
    if (value && *value)
        window_id = std::strtoull(value, nullptr, 10);
    const int cut_top = static_cast<int>(obs_data_get_int(settings, "cut_top"));
    const int cut_left = static_cast<int>(obs_data_get_int(settings, "cut_left"));
    const int cut_right = static_cast<int>(obs_data_get_int(settings, "cut_right"));
    const int source_height = static_cast<int>(obs_source_get_base_height(parent));
    filter->crop_top.store(std::max(0, cut_top));
    filter->crop_left.store(std::max(0, cut_left));
    filter->crop_right.store(std::max(0, cut_right));
    filter->crop_height.store(std::max(0, source_height));
    obs_data_release(settings);
    const uint64_t previous = filter->capture_window_id.exchange(window_id);
    {
        std::lock_guard<std::mutex> lock(filter->capture_selection_mutex);
        filter->capture_selection = value ? value : "";
    }
    if (window_id != previous || previous == 0) {
        VISION_LOG(LOG_INFO, "Face capture_window raw='%s' parsed_decimal=%llu parsed_hex=0x%llx",
             raw.c_str(), static_cast<unsigned long long>(window_id),
             static_cast<unsigned long long>(window_id));
    }
    if (window_id != previous)
        VISION_LOG(LOG_INFO, "Face XComposite capture window changed from %llu to %llu",
             static_cast<unsigned long long>(previous),
             static_cast<unsigned long long>(window_id));
}

static void x11_capture_loop(vision_filter_data *filter)
{
    x11_capture_session capture;
    while (!filter->stopping.load()) {
        std::string selection;
        {
            std::lock_guard<std::mutex> lock(filter->capture_selection_mutex);
            selection = filter->capture_selection;
        }
        if (!selection.empty()) {
            ++filter->capture_attempts;
            cv::Mat bgr;
            if (capture.capture(selection, bgr)) {
                ++filter->capture_successes;
                if (filter->capture_successes == 1 || filter->capture_successes % 30 == 0)
                    VISION_LOG(LOG_INFO, "Face XComposite capture success resolved_window=0x%llx frame=%llu size=%dx%d path=%s attempts=%llu failures=%llu",
                         static_cast<unsigned long long>(capture.resolved_window_id()),
                         static_cast<unsigned long long>(filter->capture_successes),
                         bgr.cols, bgr.rows,
                         capture.used_direct_window_fallback() ? "direct-window-fallback" : "composite-pixmap",
                         static_cast<unsigned long long>(filter->capture_attempts),
                         static_cast<unsigned long long>(filter->capture_failures));
                const int top = filter->crop_top.load();
                const int left = filter->crop_left.load();
                const int right = filter->crop_right.load();
                const int requested_height = filter->crop_height.load();
                const int crop_width = bgr.cols - left - right;
                const int crop_height = requested_height > 0
                    ? std::min(requested_height, bgr.rows - top)
                    : bgr.rows - top;
                if (left >= 0 && top >= 0 && crop_width > 0 && crop_height > 0 &&
                    (left != 0 || top != 0 || right != 0 || crop_height != bgr.rows)) {
                    if (!filter->crop_logged) {
                        VISION_LOG(LOG_INFO, "Face source crop full=%dx%d rect=%d,%d %dx%d",
                             bgr.cols, bgr.rows, left, top, crop_width, crop_height);
                        filter->crop_logged = true;
                    }
                    bgr = bgr(cv::Rect(left, top, crop_width, crop_height)).clone();
                }
                const uint32_t width = std::max(160u,
                    static_cast<uint32_t>(bgr.cols * filter->face_scale));
                const uint32_t height = std::max(90u,
                    static_cast<uint32_t>(bgr.rows * filter->face_scale));
                if (width != static_cast<uint32_t>(bgr.cols) ||
                    height != static_cast<uint32_t>(bgr.rows))
                    cv::resize(bgr, bgr, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
                submit_detection_bgr(filter, std::move(bgr));
            } else {
                ++filter->capture_failures;
                if (filter->capture_failures == 1 || filter->capture_failures % 30 == 0)
                    VISION_LOG(LOG_WARNING, "Face XComposite capture failed requested_window=%llu attempts=%llu successes=%llu failures=%llu reason='%s'",
                         static_cast<unsigned long long>(filter->capture_window_id.load()),
                         static_cast<unsigned long long>(filter->capture_attempts),
                         static_cast<unsigned long long>(filter->capture_successes),
                         static_cast<unsigned long long>(filter->capture_failures),
                         capture.last_error().c_str());
            }
        }
        const int interval = std::max(1, filter->face_interval);
        std::this_thread::sleep_for(std::chrono::milliseconds(33 * interval));
    }
}

static void submit_detection_bgr(vision_filter_data *filter, cv::Mat bgr)
{
    if (bgr.empty())
        return;

    if (!filter->first_frame_logged) {
        VISION_LOG(LOG_INFO, "Face detection received dynamic XComposite %ux%u BGR frames",
             bgr.cols, bgr.rows);
        filter->first_frame_logged = true;
    }

    filter->submitted_frames.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(filter->detection_mutex);
        filter->pending_bgr = std::move(bgr);
        filter->pending_ready = true;
    }
    filter->detection_cv.notify_one();
}


static void log_source_chain_once(vision_filter_data *filter, obs_source_t *parent)
{
    if (filter->source_chain_logged || !parent)
        return;
    filter->source_chain_logged = true;
    VISION_LOG(LOG_INFO, "Face debug parent source='%s' id='%s' base=%ux%u filters=%zu",
         obs_source_get_name(parent), obs_source_get_id(parent),
         obs_source_get_base_width(parent), obs_source_get_base_height(parent),
         obs_source_filter_count(parent));
    obs_data_t *settings = obs_source_get_settings(parent);
    if (settings) {
        VISION_LOG(LOG_INFO, "Face debug parent settings=%s", obs_data_get_json(settings));
        obs_data_release(settings);
    }
}

static void face_render(void *data, gs_effect_t *)
{
    auto *filter = static_cast<vision_filter_data *>(data);
    obs_source_t *parent = obs_filter_get_parent(filter->source);
    if (!parent || !obs_source_process_filter_begin_with_color_space(
                         filter->source, GS_RGBA, GS_CS_SRGB,
                         OBS_ALLOW_DIRECT_RENDERING))
        return;
    log_source_chain_once(filter, parent);
    refresh_capture_window(filter, parent);

    ++filter->frame_counter;
    render_faces(filter, obs_source_get_base_width(parent),
                 obs_source_get_base_height(parent));
    obs_source_process_filter_end(filter->source, filter->effect, 0, 0);
}

static void filter_destroy(void *data)
{
    auto *filter = static_cast<vision_filter_data *>(data);
    filter->stopping.store(true);
    filter->detection_cv.notify_all();
    if (filter->x11_capture_thread.joinable())
        filter->x11_capture_thread.join();
    if (filter->detector_thread.joinable())
        filter->detector_thread.join();

    obs_enter_graphics();
    if (filter->face_only) {
        gs_image_file_free(&filter->sunglasses_image);
        gs_image_file_free(&filter->pirate_hat_image);
        gs_image_file_free(&filter->clown_mask_image);
    }
    gs_effect_destroy(filter->effect);
    obs_leave_graphics();
    delete filter;
}

static void *filter_create(obs_data_t *settings, obs_source_t *source,
                           bool face_only, const char *effect_name)
{
    auto *filter = new vision_filter_data;
    filter->source = source;
    filter->face_only = face_only;

    std::string effect_relative = std::string("effects/") + effect_name;
    char *effect_path = obs_module_file(effect_relative.c_str());
    char *model_path = face_only ? obs_module_file("models/face_detection_yunet_2026may.onnx") : nullptr;
    char *sunglasses_path = face_only ? obs_module_file("images/sunglasses.png") : nullptr;
    char *pirate_hat_path = face_only ? obs_module_file("images/pirate_hat.png") : nullptr;
    char *clown_mask_path = face_only ? obs_module_file("images/clown_mask.png") : nullptr;
    char *error = nullptr;

    obs_enter_graphics();
    filter->effect = gs_effect_create_from_file(effect_path, &error);
    obs_leave_graphics();

    if (!filter->effect) {
        VISION_LOG(LOG_ERROR, "Could not load effect: %s", error ? error : effect_path);
        bfree(error);
        bfree(effect_path);
        bfree(model_path);
        bfree(sunglasses_path);
        bfree(pirate_hat_path);
        bfree(clown_mask_path);
        delete filter;
        return nullptr;
    }

    if (face_only) {
        gs_image_file_init(&filter->sunglasses_image, sunglasses_path);
        obs_enter_graphics();
        gs_image_file_init_texture(&filter->sunglasses_image);
        obs_leave_graphics();
        filter->sunglasses_image_ready = filter->sunglasses_image.loaded &&
                                          filter->sunglasses_image.texture != nullptr;
        if (!filter->sunglasses_image_ready)
            VISION_LOG(LOG_WARNING, "Could not load sunglasses overlay: %s", sunglasses_path);
        else
            VISION_LOG(LOG_INFO, "Loaded sunglasses overlay texture %ux%u",
                 filter->sunglasses_image.cx, filter->sunglasses_image.cy);
        gs_image_file_init(&filter->pirate_hat_image, pirate_hat_path);
        obs_enter_graphics();
        gs_image_file_init_texture(&filter->pirate_hat_image);
        obs_leave_graphics();
        filter->pirate_hat_image_ready = filter->pirate_hat_image.loaded &&
                                          filter->pirate_hat_image.texture != nullptr;
        if (!filter->pirate_hat_image_ready)
            VISION_LOG(LOG_WARNING, "Could not load pirate hat overlay: %s", pirate_hat_path);
        else
            VISION_LOG(LOG_INFO, "Loaded pirate hat overlay texture %ux%u",
                 filter->pirate_hat_image.cx, filter->pirate_hat_image.cy);
        gs_image_file_init(&filter->clown_mask_image, clown_mask_path);
        obs_enter_graphics();
        gs_image_file_init_texture(&filter->clown_mask_image);
        obs_leave_graphics();
        filter->clown_mask_image_ready = filter->clown_mask_image.loaded &&
                                          filter->clown_mask_image.texture != nullptr;
        if (!filter->clown_mask_image_ready)
            VISION_LOG(LOG_WARNING, "Could not load clown mask overlay: %s", clown_mask_path);
        else
            VISION_LOG(LOG_INFO, "Loaded clown mask overlay texture %ux%u",
                 filter->clown_mask_image.cx, filter->clown_mask_image.cy);
        filter->face_detector = cv::FaceDetectorYN::create(
            model_path, "", cv::Size(320, 320), 0.35f, 0.3f, 5000);
        filter->face_detector_ready = !filter->face_detector.empty();
        if (!filter->face_detector_ready)
            VISION_LOG(LOG_WARNING, "Could not load face model: %s", model_path);
        else
            VISION_LOG(LOG_INFO, "Face detection filter loaded YuNet model");
    }

    filter->threshold_param = gs_effect_get_param_by_name(filter->effect, "threshold");
    filter->strength_param = gs_effect_get_param_by_name(filter->effect, "strength");
    filter->outline_color_param = gs_effect_get_param_by_name(filter->effect, "outline_color");
    filter->fill_color_param = gs_effect_get_param_by_name(filter->effect, "fill_color");
    filter->texel_size_param = gs_effect_get_param_by_name(filter->effect, "texel_size");
    filter->color_levels_param = gs_effect_get_param_by_name(filter->effect, "color_levels");
    filter->saturation_param = gs_effect_get_param_by_name(filter->effect, "saturation");
    filter->fill_strength_param = gs_effect_get_param_by_name(filter->effect, "fill_strength");
    filter->comic_strength_param = gs_effect_get_param_by_name(filter->effect, "comic_strength");
    filter->face_count_param = gs_effect_get_param_by_name(filter->effect, "face_count");

    filter->face_box_color_param = gs_effect_get_param_by_name(filter->effect, "face_box_color");
    filter->face_box_width_param = gs_effect_get_param_by_name(filter->effect, "face_box_width");
    filter->sunglasses_box_param = gs_effect_get_param_by_name(filter->effect, "sunglasses_box");
    filter->sunglasses_angle_param = gs_effect_get_param_by_name(filter->effect, "sunglasses_angle");
    filter->sunglasses_image_param = gs_effect_get_param_by_name(filter->effect, "sunglasses_image");
    filter->pirate_hat_box_param = gs_effect_get_param_by_name(filter->effect, "pirate_hat_box");
    filter->pirate_hat_angle_param = gs_effect_get_param_by_name(filter->effect, "pirate_hat_angle");
    filter->pirate_hat_image_param = gs_effect_get_param_by_name(filter->effect, "pirate_hat_image");
    filter->clown_mask_box_param = gs_effect_get_param_by_name(filter->effect, "clown_mask_box");
    filter->clown_mask_angle_param = gs_effect_get_param_by_name(filter->effect, "clown_mask_angle");
    filter->clown_mask_image_param = gs_effect_get_param_by_name(filter->effect, "clown_mask_image");

    for (size_t i = 0; i < MAX_FACE_BOXES; ++i) {
        const std::string name = "face_box" + std::to_string(i);
        filter->face_box_params[i] = gs_effect_get_param_by_name(filter->effect, name.c_str());

    }

    bfree(effect_path);
    bfree(model_path);
    bfree(sunglasses_path);
    bfree(pirate_hat_path);
    bfree(clown_mask_path);
    if (face_only) {
        face_update(filter, settings);
        filter->detector_thread = std::thread(detector_loop, filter);
        filter->x11_capture_thread = std::thread(x11_capture_loop, filter);
    } else {
        contour_update(filter, settings);
    }
    return filter;
}

static void *contour_create(obs_data_t *settings, obs_source_t *source)
{
    return filter_create(settings, source, false, "contour.effect");
}

static void *face_create(obs_data_t *settings, obs_source_t *source)
{
    return filter_create(settings, source, true, "face.effect");
}

static obs_source_info contour_info = {};
static obs_source_info face_info = {};

} // namespace

extern "C" bool vision_filter_register(void)
{
    contour_info.id = "vision_contour_filter";
    contour_info.type = OBS_SOURCE_TYPE_FILTER;
    contour_info.output_flags = OBS_SOURCE_VIDEO;
    contour_info.get_name = contour_filter_name;
    contour_info.create = contour_create;
    contour_info.destroy = filter_destroy;
    contour_info.get_defaults = contour_defaults;
    contour_info.get_properties = contour_properties;
    contour_info.update = contour_update;
    contour_info.video_render = contour_render;

    face_info.id = "vision_face_filter";
    face_info.type = OBS_SOURCE_TYPE_FILTER;
    face_info.output_flags = OBS_SOURCE_VIDEO;
    face_info.get_name = face_filter_name;
    face_info.create = face_create;
    face_info.destroy = filter_destroy;
    face_info.get_defaults = face_defaults;
    face_info.get_properties = face_properties;
    face_info.update = face_update;
    face_info.video_render = face_render;

    obs_register_source(&contour_info);
    obs_register_source(&face_info);
    return true;
}
