include(FetchContent)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)
include(Catch)

add_executable(ocgforge_tests
    core/deck_test.cpp
    core/script_cache_test.cpp
    client/client_duel_test.cpp
    client/client_view_test.cpp
    protocol/select_unselect_test.cpp
    protocol/sum_selection_test.cpp
    duel/duel_test.cpp
    duel/clone_test.cpp
    duel/lua_memory_test.cpp
    encode/encoder_test.cpp
)
target_include_directories(ocgforge_tests PRIVATE .)
target_link_libraries(ocgforge_tests PRIVATE ygo Catch2::Catch2WithMain)
target_compile_definitions(ocgforge_tests PRIVATE
    OCGFORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    OCGFORGE_CACHE_DIR="${OCGFORGE_CACHE}"
)
catch_discover_tests(ocgforge_tests)
