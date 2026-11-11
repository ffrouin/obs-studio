#include "x11-capture.hpp"

#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/extensions/Xcomposite.h>

#include <cerrno>
#include <cstdlib>
#include <sstream>
#include <vector>

struct x11_capture_session::impl {
    Display *display = XOpenDisplay(nullptr);
};

static bool x11_error_seen = false;
static int ignore_x11_error(Display *, XErrorEvent *)
{
    x11_error_seen = true;
    return 0;
}

static unsigned char component(unsigned long pixel, unsigned long mask)
{
    if (!mask) return 0;
    int shift = 0;
    while ((mask & 1UL) == 0) { mask >>= 1; ++shift; }
    const unsigned long value = (pixel >> shift) & mask;
    return static_cast<unsigned char>((value * 255UL) / mask);
}

static bool valid_window(Display *display, Window window, XWindowAttributes &attrs)
{
    if (!window)
        return false;
    int (*old_handler)(Display *, XErrorEvent *) = XSetErrorHandler(ignore_x11_error);
    x11_error_seen = false;
    const bool ok = XGetWindowAttributes(display, window, &attrs);
    XSync(display, False);
    const bool error = x11_error_seen;
    XSetErrorHandler(old_handler);
    return ok && !error && attrs.width > 0 && attrs.height > 0;
}

static std::string window_title(Display *display, Window window)
{
    const Atom net_name = XInternAtom(display, "_NET_WM_NAME", True);
    const Atom utf8 = XInternAtom(display, "UTF8_STRING", True);
    if (net_name != None) {
        Atom actual_type = None;
        int actual_format = 0;
        unsigned long item_count = 0;
        unsigned long bytes_after = 0;
        unsigned char *data = nullptr;
        int (*old_handler)(Display *, XErrorEvent *) = XSetErrorHandler(ignore_x11_error);
        x11_error_seen = false;
        const int status = XGetWindowProperty(display, window, net_name, 0, 4096,
                                              False, utf8, &actual_type, &actual_format,
                                              &item_count, &bytes_after, &data);
        XSync(display, False);
        const bool error = x11_error_seen;
        XSetErrorHandler(old_handler);
        if (!error && status == Success && data) {
            std::string result(reinterpret_cast<char *>(data), item_count);
            XFree(data);
            return result;
        }
        if (data) XFree(data);
    }

    char *name = nullptr;
    std::string result;
    if (XFetchName(display, window, &name) && name) {
        result = name;
        XFree(name);
    }
    return result;
}

static bool window_matches(Display *display, Window window,
                           const std::string &wanted_title,
                           const std::string &wanted_class)
{
    XClassHint hint{};
    const bool has_class = XGetClassHint(display, window, &hint);
    const std::string title = window_title(display, window);
    const std::string res_name = has_class && hint.res_name ? hint.res_name : "";
    const std::string res_class = has_class && hint.res_class ? hint.res_class : "";
    if (hint.res_name) XFree(hint.res_name);
    if (hint.res_class) XFree(hint.res_class);
    const bool title_ok = wanted_title.empty() || title == wanted_title;
    const bool class_ok = wanted_class.empty() || res_name == wanted_class || res_class == wanted_class;
    return title_ok && class_ok;
}

static std::vector<Window> client_windows(Display *display)
{
    std::vector<Window> result;
    const Atom client_list = XInternAtom(display, "_NET_CLIENT_LIST", True);
    if (client_list == None)
        return result;

    Atom actual_type = None;
    int actual_format = 0;
    unsigned long item_count = 0;
    unsigned long bytes_after = 0;
    unsigned char *data = nullptr;
    int (*old_handler)(Display *, XErrorEvent *) = XSetErrorHandler(ignore_x11_error);
    x11_error_seen = false;
    const int status = XGetWindowProperty(display, DefaultRootWindow(display), client_list,
                                          0, 4096, False, XA_WINDOW, &actual_type,
                                          &actual_format, &item_count, &bytes_after, &data);
    XSync(display, False);
    const bool error = x11_error_seen;
    XSetErrorHandler(old_handler);
    if (!error && status == Success && actual_type == XA_WINDOW && actual_format == 32 && data) {
        const Window *windows = reinterpret_cast<const Window *>(data);
        result.assign(windows, windows + item_count);
    }
    if (data) XFree(data);
    return result;
}

static Window resolve_selection(Display *display, const std::string &encoded,
                                uint64_t &numeric_id, std::string &wanted_title,
                                std::string &wanted_class)
{
    std::istringstream input(encoded);
    std::string id_text;
    std::getline(input, id_text);
    std::getline(input, wanted_title);
    std::getline(input, wanted_class);
    if (!id_text.empty() && id_text.back() == '\r') id_text.pop_back();
    if (!wanted_title.empty() && wanted_title.back() == '\r') wanted_title.pop_back();
    if (!wanted_class.empty() && wanted_class.back() == '\r') wanted_class.pop_back();
    errno = 0;
    numeric_id = id_text.empty() ? 0 : std::strtoull(id_text.c_str(), nullptr, 10);

    XWindowAttributes attrs{};
    if (numeric_id && valid_window(display, static_cast<Window>(numeric_id), attrs))
        return static_cast<Window>(numeric_id);

    std::vector<Window> candidates = client_windows(display);
    if (candidates.empty()) {
        Window root = DefaultRootWindow(display);
        Window parent = 0;
        Window *children = nullptr;
        unsigned int count = 0;
        int (*old_handler)(Display *, XErrorEvent *) = XSetErrorHandler(ignore_x11_error);
        x11_error_seen = false;
        const bool tree_ok = XQueryTree(display, root, &root, &parent, &children, &count);
        XSync(display, False);
        const bool tree_error = x11_error_seen;
        XSetErrorHandler(old_handler);
        if (tree_ok && !tree_error && children)
            candidates.assign(children, children + count);
        if (children) XFree(children);
    }

    for (const Window candidate : candidates) {
        XWindowAttributes candidate_attrs{};
        if (valid_window(display, candidate, candidate_attrs) &&
            window_matches(display, candidate, wanted_title, wanted_class))
            return candidate;
    }
    return 0;
}

x11_capture_session::x11_capture_session() : state_(new impl) {}
x11_capture_session::~x11_capture_session()
{
    if (state_->display) XCloseDisplay(state_->display);
    delete state_;
}

bool x11_capture_session::capture(const std::string &encoded_selection, cv::Mat &bgr)
{
    last_error_.clear();
    resolved_window_id_ = 0;
    used_direct_window_fallback_ = false;
    if (!state_->display) {
        last_error_ = "XOpenDisplay failed";
        return false;
    }
    if (encoded_selection.empty()) {
        last_error_ = "encoded window selection is empty";
        return false;
    }

    uint64_t numeric_id = 0;
    std::string wanted_title;
    std::string wanted_class;
    const Window window = resolve_selection(state_->display, encoded_selection,
                                            numeric_id, wanted_title, wanted_class);
    if (!window) {
        last_error_ = "numeric window invalid and _NET_CLIENT_LIST title/class fallback found no window";
        return false;
    }
    resolved_window_id_ = static_cast<uint64_t>(window);

    XWindowAttributes attrs{};
    if (!valid_window(state_->display, window, attrs)) {
        last_error_ = "resolved window became invalid";
        return false;
    }

    Pixmap pixmap = 0;
    int (*old_handler)(Display *, XErrorEvent *) = XSetErrorHandler(ignore_x11_error);
    x11_error_seen = false;
    pixmap = XCompositeNameWindowPixmap(state_->display, window);
    XSync(state_->display, False);
    const bool composite_error = x11_error_seen;
    const bool composite_ok = !composite_error && pixmap != 0;
    XSetErrorHandler(old_handler);

    old_handler = XSetErrorHandler(ignore_x11_error);
    x11_error_seen = false;
    XImage *image = composite_ok
        ? XGetImage(state_->display, pixmap, 0, 0, attrs.width, attrs.height, AllPlanes, ZPixmap)
        : XGetImage(state_->display, window, 0, 0, attrs.width, attrs.height, AllPlanes, ZPixmap);
    XSync(state_->display, False);
    const bool image_error = x11_error_seen;
    bool image_ok = !image_error && image != nullptr;
    XSetErrorHandler(old_handler);
    if (composite_ok)
        XFreePixmap(state_->display, pixmap);
    if (!image_ok) {
        if (image) XDestroyImage(image);
        last_error_ = composite_ok
            ? "XGetImage on named composite pixmap failed"
            : (composite_error ? "composite pixmap unavailable; XGetImage fallback failed"
                                : "XGetImage on window failed");
        return false;
    }

    if (composite_ok) {
        bool blank = true;
        for (int y = 0; y < image->height && blank; y += 8) {
            for (int x = 0; x < image->width; x += 8) {
                const unsigned long pixel = XGetPixel(image, x, y);
                if (component(pixel, image->red_mask) != 0 ||
                    component(pixel, image->green_mask) != 0 ||
                    component(pixel, image->blue_mask) != 0) {
                    blank = false;
                    break;
                }
            }
        }
        if (blank) {
            XDestroyImage(image);
            old_handler = XSetErrorHandler(ignore_x11_error);
            x11_error_seen = false;
            image = XGetImage(state_->display, window, 0, 0, attrs.width, attrs.height,
                              AllPlanes, ZPixmap);
            XSync(state_->display, False);
            const bool fallback_error = x11_error_seen;
            image_ok = !fallback_error && image != nullptr;
            XSetErrorHandler(old_handler);
            used_direct_window_fallback_ = true;
            if (!image_ok) {
                if (image) XDestroyImage(image);
                last_error_ = "composite pixmap was blank and direct window fallback failed";
                return false;
            }
        }
    }

    bgr.create(image->height, image->width, CV_8UC3);
    for (int y = 0; y < image->height; ++y) {
        cv::Vec3b *row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image->width; ++x) {
            const unsigned long pixel = XGetPixel(image, x, y);
            row[x] = cv::Vec3b(component(pixel, image->blue_mask),
                               component(pixel, image->green_mask),
                               component(pixel, image->red_mask));
        }
    }
    XDestroyImage(image);
    return true;
}
