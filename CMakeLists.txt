cmake_minimum_required(VERSION 3.16)
project(obs-vision-filter VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBOBS REQUIRED libobs)

pkg_check_modules(X11_CAPTURE REQUIRED x11 xcomposite xext)

find_path(OPENCV_INCLUDE_DIR opencv2/core.hpp
    PATHS /usr/include/opencv5 /usr/local/include/opencv5)
find_library(OPENCV_CORE_LIBRARY NAMES opencv_core PATHS /usr/lib /usr/local/lib)
find_library(OPENCV_DNN_LIBRARY NAMES opencv_dnn PATHS /usr/lib /usr/local/lib)
find_library(OPENCV_IMGPROC_LIBRARY NAMES opencv_imgproc PATHS /usr/lib /usr/local/lib)
find_library(OPENCV_OBJDETECT_LIBRARY NAMES opencv_objdetect PATHS /usr/lib /usr/local/lib)
find_library(OPENCV_VIDEO_LIBRARY NAMES opencv_video PATHS /usr/lib /usr/local/lib)

if (NOT OPENCV_INCLUDE_DIR OR NOT OPENCV_CORE_LIBRARY OR NOT OPENCV_DNN_LIBRARY OR
    NOT OPENCV_IMGPROC_LIBRARY OR NOT OPENCV_OBJDETECT_LIBRARY OR
    NOT OPENCV_VIDEO_LIBRARY)
    message(FATAL_ERROR "OpenCV core, imgproc, and objdetect development files are required")
endif()

add_library(obs-vision-filter MODULE
    src/plugin-main.cpp
    src/vision-filter.cpp
    src/x11-capture.cpp
)

target_include_directories(obs-vision-filter PRIVATE
    ${LIBOBS_INCLUDEDIR}
    ${OPENCV_INCLUDE_DIR}
)
target_link_directories(obs-vision-filter PRIVATE ${LIBOBS_LIBRARY_DIRS})
target_link_libraries(obs-vision-filter PRIVATE
    ${LIBOBS_LIBRARIES}
    ${OPENCV_CORE_LIBRARY}
    ${OPENCV_DNN_LIBRARY}
    ${OPENCV_IMGPROC_LIBRARY}
    ${OPENCV_OBJDETECT_LIBRARY}
    ${OPENCV_VIDEO_LIBRARY}
    ${X11_CAPTURE_LIBRARIES}
)
target_compile_options(obs-vision-filter PRIVATE -Wall -Wextra -Wpedantic)

add_executable(xcomposite-probe xcomposite_probe.cpp)
target_include_directories(xcomposite-probe PRIVATE ${X11_CAPTURE_INCLUDE_DIRS})
target_link_directories(xcomposite-probe PRIVATE ${X11_CAPTURE_LIBRARY_DIRS})
target_link_libraries(xcomposite-probe PRIVATE ${X11_CAPTURE_LIBRARIES})
target_compile_options(xcomposite-probe PRIVATE -Wall -Wextra -Wpedantic)

set_target_properties(obs-vision-filter PROPERTIES
    PREFIX ""
    OUTPUT_NAME "obs-vision-filter"
)

include(GNUInstallDirs)
install(TARGETS obs-vision-filter
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/obs-plugins
)
install(DIRECTORY data/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/obs/obs-plugins/obs-vision-filter
)
