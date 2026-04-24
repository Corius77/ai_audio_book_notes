cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(LSTMPlugin VERSION 0.1.0)

set(LIBTORCH_DIR "C:/Users/sking/OneDrive/Dokumenty/ai_audio_book_notes/libtorch")
set(CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
find_package(Torch REQUIRED)

# JUCE
add_subdirectory(../JUCE JUCE)

# AudioPluginHost (built alongside the plugin for debugging)
add_subdirectory(../JUCE/extras/AudioPluginHost AudioPluginHost)

# ---------------------------------------------------------------------------
# Plugin target
# ---------------------------------------------------------------------------
juce_add_plugin(LSTMPlugin
    COMPANY_NAME                    "Corius"
    PLUGIN_MANUFACTURER_CODE        CORS
    PLUGIN_CODE                     LSTM
    FORMATS                         VST3
    PRODUCT_NAME                    "LSTM Plugin"
    IS_SYNTH                        FALSE
    NEEDS_MIDI_INPUT                FALSE
    NEEDS_MIDI_OUTPUT               FALSE
    IS_MIDI_EFFECT                  FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS     FALSE
)

juce_generate_juce_header(LSTMPlugin)

target_sources(LSTMPlugin PRIVATE
    source/PluginProcessor.cpp
    source/PluginProcessor.h
    source/PluginEditor.cpp
    source/PluginEditor.h
)

target_compile_definitions(LSTMPlugin PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_compile_options(LSTMPlugin PRIVATE ${TORCH_CXX_FLAGS})

target_link_libraries(LSTMPlugin
    PRIVATE
        juce::juce_audio_utils
        "${TORCH_LIBRARIES}"
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

set_property(TARGET LSTMPlugin PROPERTY CXX_STANDARD 20)

add_custom_command(TARGET LSTMPlugin_VST3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/my_lstm_with_state.pt"
            "$<TARGET_FILE_DIR:LSTMPlugin_VST3>/my_lstm_with_state.pt"
)

# Copy LibTorch DLLs into the VST3 bundle so AudioPluginHost can load the plugin
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET LSTMPlugin_VST3 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:LSTMPlugin_VST3>
        COMMENT "Copying LibTorch DLLs into VST3 bundle"
    )
endif()
