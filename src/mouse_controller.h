#pragma once
#include <windows.h>

class MouseController {
public:
    MouseController();
    ~MouseController();

    bool Load(const char *dllPath);
    bool IsLoaded() const { return m_loaded; }
    void MoveRelative(int dx, int dy);

private:
    HINSTANCE m_dll = nullptr;
    bool m_loaded = false;
    using MoveRFn = void (*)(int, int);
    MoveRFn m_moveR = nullptr;
};
