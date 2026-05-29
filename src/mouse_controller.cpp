#include "mouse_controller.h"
#include <windows.h>

MouseController::MouseController() = default;

MouseController::~MouseController() {
    if (m_dll) FreeLibrary(m_dll);
}

bool MouseController::Load(const char *dllPath) {
    m_dll = LoadLibraryA(dllPath);
    if (!m_dll) return false;

    using OpenDeviceFn = int (*)();
    auto openDev = (OpenDeviceFn)GetProcAddress(m_dll, "OpenDevice");
    if (!openDev || openDev() == 0) {
        FreeLibrary(m_dll);
        m_dll = nullptr;
        return false;
    }

    m_moveR = (MoveRFn)GetProcAddress(m_dll, "MoveR");
    if (!m_moveR) {
        FreeLibrary(m_dll);
        m_dll = nullptr;
        return false;
    }

    m_loaded = true;
    return true;
}

void MouseController::MoveRelative(int dx, int dy) {
    if (m_moveR) m_moveR(dx, dy);
}
