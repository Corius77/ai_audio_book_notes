cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(minimal-libtorch)

set(CMAKE_PREFIX_PATH "C:/Users/sking/OneDrive/Dokumenty/NeuralAmp/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(minimal-libtorch main.cpp
    AudioFile/AudioFile.h)

target_link_libraries(minimal-libtorch "${TORCH_LIBRARIES}")
set_property(TARGET minimal-libtorch PROPERTY CXX_STANDARD 20)

# Dodajemy komendę kopiowania pliku PO zbudowaniu aplikacji
add_custom_command(
    TARGET minimal-libtorch POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/my_lstm.pt"
            "$<TARGET_FILE_DIR:minimal-libtorch>/my_lstm.pt"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/sine_400_16bit.wav"
            "$<TARGET_FILE_DIR:minimal-libtorch>/sine_400_16bit.wav"
    COMMENT "Kopiowanie modelu do katalogu z plikiem wykonywalnym"
)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET minimal-libtorch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:minimal-libtorch>)
endif (MSVC)