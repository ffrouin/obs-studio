#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/Xcomposite.h>
#include <X11/Xatom.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static bool x_error_seen = false;

static int ignored_x_error(Display *, XErrorEvent *)
{
    x_error_seen = true;
    return 0;
}

static std::string window_title(Display *display, Window window)
{
    XTextProperty prop{};
    if (XGetWMName(display, window, &prop) && prop.value) {
        std::string result(reinterpret_cast<char *>(prop.value));
        XFree(prop.value);
        return result;
    }
    return {};
}

static Window find_window(Display *display, Window root, const std::string &needle)
{
    const std::string title = window_title(display, root);
    if (!title.empty() && (needle.empty() || title.find(needle) != std::string::npos))
        return root;

    Window root_return = 0, parent_return = 0;
    Window *children = nullptr;
    unsigned int count = 0;
    if (!XQueryTree(display, root, &root_return, &parent_return, &children, &count))
        return 0;
    Window found = 0;
    for (unsigned int i = 0; i < count && !found; ++i)
        found = find_window(display, children[i], needle);
    if (children)
        XFree(children);
    return found;
}

static unsigned char component(unsigned long pixel, unsigned long mask)
{
    if (!mask) return 0;
    int shift = 0;
    while ((mask & 1UL) == 0) { mask >>= 1; ++shift; }
    const unsigned long value = (pixel & (mask << shift)) >> shift;
    return static_cast<unsigned char>((value * 255UL) / mask);
}

static bool write_ppm_image(XImage *image, const std::string &path)
{
    if (!image) return false;
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P6\n" << image->width << " " << image->height << "\n255\n";
    std::vector<unsigned char> row(static_cast<size_t>(image->width) * 3);
    for (int y = 0; y < image->height; ++y) {
        for (int x = 0; x < image->width; ++x) {
            const unsigned long pixel = XGetPixel(image, x, y);
            row[static_cast<size_t>(x) * 3 + 0] = component(pixel, image->red_mask);
            row[static_cast<size_t>(x) * 3 + 1] = component(pixel, image->green_mask);
            row[static_cast<size_t>(x) * 3 + 2] = component(pixel, image->blue_mask);
        }
        out.write(reinterpret_cast<const char *>(row.data()), static_cast<std::streamsize>(row.size()));
    }
    return static_cast<bool>(out);
}

static bool write_ppm(Display *display, Window window, Pixmap pixmap, const std::string &path)
{
    XWindowAttributes attrs{};
    if (!XGetWindowAttributes(display, window, &attrs)) return false;
    XImage *image = nullptr;
    XSetErrorHandler(ignored_x_error);
    x_error_seen = false;
    image = pixmap
        ? XGetImage(display, pixmap, 0, 0, attrs.width, attrs.height, AllPlanes, ZPixmap)
        : XGetImage(display, window, 0, 0, attrs.width, attrs.height, AllPlanes, ZPixmap);
    XSync(display, False);
    const bool image_ok = !x_error_seen && image != nullptr;
    XSetErrorHandler(nullptr);
    if (!image_ok && image) XDestroyImage(image);
    if (!image_ok) return false;
    const bool ok = write_ppm_image(image, path);
    XDestroyImage(image);
    return ok;
}

int main(int argc, char **argv)
{
    const std::string title = argc > 1 ? argv[1] : "";
    const std::string output = argc > 2 ? argv[2] : "xcomposite-probe.ppm";
    Display *display = XOpenDisplay(nullptr);
    if (!display) { std::cerr << "cannot open X display\n"; return 2; }
    const Window root = DefaultRootWindow(display);
    const Window window = find_window(display, root, title);
    if (!window) { std::cerr << "window not found: " << title << "\n"; XCloseDisplay(display); return 3; }

    XWindowAttributes attrs{};
    XGetWindowAttributes(display, window, &attrs);
    XSetErrorHandler(ignored_x_error);
    x_error_seen = false;
    const Pixmap pixmap = XCompositeNameWindowPixmap(display, window);
    XSync(display, False);
    const bool composite_ok = !x_error_seen && pixmap != 0;
    XSetErrorHandler(nullptr);
    if (!composite_ok)
        std::cerr << "XCompositeNameWindowPixmap unavailable; falling back to visible XGetImage capture\n";
    std::cout << "window=0x" << std::hex << window << std::dec
              << " title='" << window_title(display, window) << "' size="
              << attrs.width << "x" << attrs.height << "\n";
    XSync(display, False);
    const bool ok = write_ppm(display, window, composite_ok ? pixmap : 0, output);
    if (composite_ok)
        XFreePixmap(display, pixmap);
    XCloseDisplay(display);
    std::cout << (ok ? "wrote " : "failed ") << output << "\n";
    return ok ? 0 : 5;
}
