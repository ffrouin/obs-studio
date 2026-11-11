#include <obs/obs-module.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-vision-filter", "en-US")

extern "C" bool vision_filter_register(void);

bool obs_module_load(void)
{
    return vision_filter_register();
}

void obs_module_unload(void)
{
}
