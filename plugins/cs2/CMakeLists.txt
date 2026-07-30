cmake_minimum_required(VERSION 3.20)
project(aegis_cs2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(AEGIS_TOOLS_ONLY "Build SDK-free protocol tools only" OFF)
option(AEGIS_WITH_METAMOD "Build MetaMod plugin (needs HL2SDKCS2 + MMSOURCE112)" OFF)

set(AEGIS_COMMON_SOURCES
  src/encode.cpp
  src/frames.cpp
  src/net_client.cpp
  src/salt.cpp
  src/vis_worker.cpp
  src/aegis_mm.cpp
)

add_library(aegis_protocol STATIC
  src/encode.cpp
  src/frames.cpp
  src/net_client.cpp
  src/salt.cpp
  src/vis_worker.cpp
  src/aegis_mm.cpp
)
target_include_directories(aegis_protocol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(WIN32)
  target_link_libraries(aegis_protocol PUBLIC ws2_32)
endif()

add_executable(encode_smoke tools/encode_smoke.cpp)
target_link_libraries(encode_smoke PRIVATE aegis_protocol)

if(AEGIS_WITH_METAMOD)
  if(NOT DEFINED ENV{HL2SDKCS2} AND NOT HL2SDKCS2)
    message(FATAL_ERROR "Set HL2SDKCS2 to the hl2sdk-cs2 root")
  endif()
  if(NOT DEFINED ENV{MMSOURCE112} AND NOT MMSOURCE112)
    message(FATAL_ERROR "Set MMSOURCE112 to the metamod-source root")
  endif()
  if(NOT HL2SDKCS2)
    set(HL2SDKCS2 $ENV{HL2SDKCS2})
  endif()
  if(NOT MMSOURCE112)
    set(MMSOURCE112 $ENV{MMSOURCE112})
  endif()

  add_library(aegis SHARED ${AEGIS_COMMON_SOURCES})
  target_compile_definitions(aegis PRIVATE AEGIS_WITH_METAMOD)
  target_include_directories(aegis PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${MMSOURCE112}/core
    ${MMSOURCE112}/core/sourcehook
    ${HL2SDKCS2}/public
    ${HL2SDKCS2}/public/engine
    ${HL2SDKCS2}/public/tier0
    ${HL2SDKCS2}/public/tier1
    ${HL2SDKCS2}/game/shared
  )
  if(WIN32)
    target_link_libraries(aegis PRIVATE ws2_32)
  endif()
  # Full link line against tier0/tier1/interfaces is SDK-version specific —
  # fill in once your hl2sdk-cs2 tree is checked out.
endif()

install(FILES addons/metamod/aegis.vdf DESTINATION addons/metamod)
