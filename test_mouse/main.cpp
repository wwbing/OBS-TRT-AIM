#include <windows.h>
#include <cstdio>

int main() {
    // Load DLL
    const char *dllPath = "D:/code/obs-plugintemplate/data/ddll64.dll";
    HINSTANCE dll = LoadLibraryA(dllPath);
    if (!dll) { printf("FAIL: LoadLibrary '%s' error=%lu\n", dllPath, GetLastError()); return 1; }
    printf("OK: DLL loaded\n");

    // Open device
    using OpenDeviceFn = int (*)();
    auto openDev = (OpenDeviceFn)GetProcAddress(dll, "OpenDevice");
    if (!openDev) { printf("FAIL: OpenDevice not found\n"); return 1; }
    if (openDev() == 0) { printf("FAIL: OpenDevice returned 0\n"); return 1; }
    printf("OK: Device opened\n");

    // Check MoveR
    using MoveRFn = void (*)(int, int);
    auto moveR = (MoveRFn)GetProcAddress(dll, "MoveR");
    if (!moveR) { printf("FAIL: MoveR not found\n"); return 1; }
    printf("OK: MoveR found\n");

    // Test relative move: move right 100px then down 100px
    printf("Moving (100, 0) in 1s...\n");
    Sleep(1000);
    moveR(100, 0);
    printf("Done\n");

    printf("Moving (0, 100) in 1s...\n");
    Sleep(1000);
    moveR(0, 100);
    printf("Done\n");

    printf("Moving (-100, -100) in 1s...\n");
    Sleep(1000);
    moveR(-100, -100);
    printf("Done\n");

    FreeLibrary(dll);
    printf("All tests passed. DLL unloaded.\n");
    return 0;
}
