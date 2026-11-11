#include <obs/obs-module.h>
#include <obs/obs.h>
#include <obs/graphics/graphics.h>

#include <cstring>
#include <mutex>
#include <string>

namespace {
struct vision_source_data {
    obs_source_t *source = nullptr;
    std::mutex source_mutex;
    gs_texrender_t *texrender = nullptr;
    std::string input_name;
};

static const char *source_name(void *)
{
    return "Vision source tap";
}

static void resolve_input(vision_source_data *data)
{
    if (data->source) {
        obs_source_release(data->source);
        data->source = nullptr;
    }
    if (!data->input_name.empty())
        data->source = obs_get_source_by_name(data->input_name.c_str());
    blog(LOG_INFO, "Vision source tap input='%s' resolved=%s",
         data->input_name.c_str(), data->source ? "yes" : "no");
}

static void *source_create(obs_data_t *settings, obs_source_t *)
{
    auto *data = new vision_source_data;
    data->input_name = obs_data_get_string(settings, "input_source");
    resolve_input(data);
    obs_enter_graphics();
    data->texrender = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
    obs_leave_graphics();
    return data;
}

static void source_destroy(void *opaque)
{
    auto *data = static_cast<vision_source_data *>(opaque);
    {
        std::lock_guard<std::mutex> lock(data->source_mutex);
        if (data->source) {
            obs_source_release(data->source);
            data->source = nullptr;
        }
    }
    obs_enter_graphics();
    gs_texrender_destroy(data->texrender);
    obs_leave_graphics();
    delete data;
}

static void source_update(void *opaque, obs_data_t *settings)
{
    auto *data = static_cast<vision_source_data *>(opaque);
    const std::string next = obs_data_get_string(settings, "input_source");
    std::lock_guard<std::mutex> lock(data->source_mutex);
    if (next != data->input_name) {
        data->input_name = next;
        resolve_input(data);
    }
}

static uint32_t source_width(void *opaque)
{
    auto *data = static_cast<vision_source_data *>(opaque);
    std::lock_guard<std::mutex> lock(data->source_mutex);
    return data->source ? obs_source_get_width(data->source) : 0;
}

static uint32_t source_height(void *opaque)
{
    auto *data = static_cast<vision_source_data *>(opaque);
    std::lock_guard<std::mutex> lock(data->source_mutex);
    return data->source ? obs_source_get_height(data->source) : 0;
}

static void source_video_render(void *opaque, gs_effect_t *)
{
    auto *data = static_cast<vision_source_data *>(opaque);
    obs_source_t *source = nullptr;
    {
        std::lock_guard<std::mutex> lock(data->source_mutex);
        if (data->source)
            source = obs_source_get_ref(data->source);
    }
    if (!source || !data->texrender) {
        obs_source_release(source);
        return;
    }
    const uint32_t width = obs_source_get_width(source);
    const uint32_t height = obs_source_get_height(source);
    if (!width || !height || !gs_texrender_begin(data->texrender, width, height)) {
        obs_source_release(source);
        return;
    }
    gs_clear(GS_CLEAR_COLOR, nullptr, 0.0f, 0);
    obs_source_video_render(source);
    gs_texrender_end(data->texrender);
    obs_source_release(source);
    gs_draw_sprite(gs_texrender_get_texture(data->texrender), 0, width, height);
}

static bool add_xcomposite_source(void *param, obs_source_t *source)
{
    if (std::strcmp(obs_source_get_id(source), "xcomposite_input") != 0)
        return true;
    auto *property = static_cast<obs_property_t *>(param);
    const char *name = obs_source_get_name(source);
    if (name && *name)
        obs_property_list_add_string(property, name, name);
    return true;
}

static obs_properties_t *source_properties(void *)
{
    obs_properties_t *props = obs_properties_create();
    obs_property_t *list = obs_properties_add_list(
        props, "input_source", "XComposite source",
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
    obs_enum_sources(add_xcomposite_source, list);
    return props;
}

static void source_defaults(obs_data_t *settings)
{
    obs_data_set_string(settings, "input_source", "");
}

} // namespace

extern "C" bool vision_source_register(void)
{
    static obs_source_info info = {};
    info.id = "vision_source_tap";
    info.type = OBS_SOURCE_TYPE_INPUT;
    info.output_flags = OBS_SOURCE_VIDEO;
    info.get_name = source_name;
    info.create = source_create;
    info.destroy = source_destroy;
    info.update = source_update;
    info.get_width = source_width;
    info.get_height = source_height;
    info.get_properties = source_properties;
    info.get_defaults = source_defaults;
    info.video_render = source_video_render;
    obs_register_source(&info);
    return true;
}
