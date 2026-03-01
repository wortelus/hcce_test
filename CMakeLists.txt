cmake_minimum_required(VERSION 4.1)
project(hcce_test)

set(CMAKE_CXX_STANDARD 20)

# Najdi OpenCV
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

set(ONNX_PKG "${CMAKE_SOURCE_DIR}/packages/Microsoft.ML.OnnxRuntime.1.24.2")
# .dll jsou zde (pro runtime)
link_directories("${ONNX_PKG}/runtimes/win-x64/native")
# .lib je pravděpodobně zde (pro linkování)
link_directories("${ONNX_PKG}/build/native/lib/x64/Release")
# Include cesta (hlavičkové soubory)
include_directories("${ONNX_PKG}/build/native/include")
# Lib cesta (knihovna pro linkování)
link_directories("${ONNX_PKG}/runtimes/win-x64/native")

add_executable(HccePoseRunner src/main.cpp
        src/types.h
        src/preprocessing.cpp
        src/preprocessing.h
        src/yolodetector.cpp
        src/yolodetector.h
        src/croptransform.cpp
        src/croptransform.h
        src/hcceposeestimator.cpp
        src/hcceposeestimator.h
        src/hccedecoder.cpp
        src/hccedecoder.h
        src/ultradensesampler.cpp
        src/ultradensesampler.h
        src/pnpsolver.cpp
        src/pnpsolver.h
        src/visualizer.cpp
        src/visualizer.h
)


target_link_libraries(HccePoseRunner
        ${OpenCV_LIBS}
        onnxruntime
)

# OpenCV DLL
add_custom_command(TARGET HccePoseRunner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:opencv_world>
        $<TARGET_FILE_DIR:HccePoseRunner>
)

# ONNX Runtime DLL
add_custom_command(TARGET HccePoseRunner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNX_PKG}/runtimes/win-x64/native/onnxruntime.dll"
        $<TARGET_FILE_DIR:HccePoseRunner>
)