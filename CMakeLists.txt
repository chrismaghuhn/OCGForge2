cmake_minimum_required(VERSION 3.24)
project(OCGForge LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # the static libraries also go into the Python module

option(OCGFORGE_BUILD_TESTS "Build the test suite" ON)
option(OCGFORGE_BUILD_PYTHON "Build the ocgforge._core Python module" OFF)

# --- Pinned third-party inputs ------------------------------------------------
# Fetches ocgcore, CardScripts and BabelCDB into .cache/ (see third_party/deps.json),
# applies our ocgcore patches and writes .cache/card_data.tsv for the decks.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(OCGFORGE_CACHE "${CMAKE_SOURCE_DIR}/.cache")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/setup_deps.py" --cache "${OCGFORGE_CACHE}"
    RESULT_VARIABLE setup_result
)
if(NOT setup_result EQUAL 0)
    message(FATAL_ERROR "tools/setup_deps.py failed")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/third_party/deps.json" "${CMAKE_SOURCE_DIR}/tools/setup_deps.py")

# --- ocgcore (rules engine + bundled Lua, compiled as C++) --------------------
set(OCGCORE_DIR "${OCGFORGE_CACHE}/ocgcore")
file(GLOB OCGCORE_SOURCES "${OCGCORE_DIR}/*.cpp")
# Only the Lua modules ocgcore needs (no os, package or debug libraries).
set(LUA_MODULES lapi lauxlib lbaselib lcode lctype ldebug ldo ldump lfunc lgc liolib llex
    lmathlib lmem lobject lopcodes lparser lstate lstring lstrlib ltable ltablib ltm lundump lvm lzio)
list(TRANSFORM LUA_MODULES PREPEND "${OCGCORE_DIR}/lua/src/" OUTPUT_VARIABLE LUA_SOURCES)
list(TRANSFORM LUA_SOURCES APPEND ".c")
set_source_files_properties(${LUA_SOURCES} PROPERTIES LANGUAGE CXX)

add_library(ocgcore STATIC ${OCGCORE_SOURCES} ${LUA_SOURCES})
target_include_directories(ocgcore SYSTEM PUBLIC "${OCGCORE_DIR}" "${OCGCORE_DIR}/lua/src")
target_compile_options(ocgcore PRIVATE "-include${OCGCORE_DIR}/lua/luaconf-customize.h" -w)

# --- OCGForge ----------------------------------------------------------------
add_library(ygo STATIC
    src/util/sha256.cpp
    src/core/card_data_reader.cpp
    src/core/core_host.cpp
    src/core/deck.cpp
    src/core/script_reader.cpp
    src/protocol/continuation.cpp
    src/protocol/message_decoder.cpp
    src/protocol/response_builder.cpp
    src/observation/card_projection.cpp
    src/observation/decision_integration.cpp
    src/observation/event_projection.cpp
    src/observation/observation_builder.cpp
    src/observation/observation_session.cpp
    src/observation/query_decoder.cpp
    src/observation/serialization.cpp
    src/observation/zone_projection.cpp
    src/client/client_duel.cpp
    src/client/client_view.cpp
    src/duel/duel.cpp
    src/encode/encoder.cpp
    src/encode/vector_env.cpp
)
target_include_directories(ygo PUBLIC include PRIVATE src/core src/observation src/client)
target_link_libraries(ygo PUBLIC ocgcore)
target_compile_options(ygo PRIVATE -Wall -Wextra -Wpedantic)
find_package(Threads REQUIRED)
target_link_libraries(ygo PUBLIC Threads::Threads)

if(OCGFORGE_BUILD_PYTHON)
    # Build against the interpreter that will import it, e.g. -DPython_EXECUTABLE=.venv/bin/python
    find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
    include(FetchContent)
    FetchContent_Declare(nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.13.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nanobind)
    nanobind_add_module(_core src/python/module.cpp)
    target_link_libraries(_core PRIVATE ygo)
    # Next to the Python sources, so an editable install imports it directly.
    set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/src/ocgforge")
endif()

if(OCGFORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
