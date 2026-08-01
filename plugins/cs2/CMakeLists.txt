cmake_minimum_required(VERSION 3.20)
project(aegis_cs2 LANGUAGES CXX)

cmake_policy(SET CMP0091 NEW)  # MSVC_RUNTIME_LIBRARY honoured

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
  src/sdk_bridge.cpp
  src/aegis_mm.cpp
  src/module_scan.cpp
  src/schema_offset.cpp
)

add_library(aegis_protocol STATIC
  src/encode.cpp
  src/frames.cpp
  src/net_client.cpp
  src/salt.cpp
  src/vis_worker.cpp
  src/sdk_bridge.cpp
  src/aegis_mm.cpp
  src/module_scan.cpp
  src/schema_offset.cpp
)
target_include_directories(aegis_protocol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(WIN32)
  target_link_libraries(aegis_protocol PUBLIC ws2_32 bcrypt)
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
  # hl2sdk static libs are built against the static CRT; mixing them with /MD
  # produces LNK4098 and corrupted pointers once the engine calls into us.
  set_property(TARGET aegis PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
  target_compile_definitions(aegis PRIVATE
    AEGIS_WITH_METAMOD
    META_IS_SOURCE2
    SOURCE_ENGINE=25
    SE_CS2=25
    COMPILER_MSVC
    COMPILER_MSVC64
    WIN32
    _WINDOWS
    _CRT_SECURE_NO_WARNINGS
    _CRT_NONSTDC_NO_DEPRECATE
  )
  set(AEGIS_PROTO_GEN ${CMAKE_CURRENT_BINARY_DIR}/generated)
  file(MAKE_DIRECTORY ${AEGIS_PROTO_GEN})
  if(EXISTS ${HL2SDKCS2}/devtools/bin/protoc.exe AND EXISTS ${HL2SDKCS2}/common/network_connection.proto)
    execute_process(
      COMMAND ${HL2SDKCS2}/devtools/bin/protoc.exe
        --proto_path=${HL2SDKCS2}/common
        --proto_path=${HL2SDKCS2}/thirdparty/protobuf-3.21.8/src
        --cpp_out=${AEGIS_PROTO_GEN}
        ${HL2SDKCS2}/common/network_connection.proto
      RESULT_VARIABLE _aegis_protoc_rc
    )
    if(NOT _aegis_protoc_rc EQUAL 0)
      message(WARNING "protoc failed for network_connection.proto (rc=${_aegis_protoc_rc})")
    endif()
  endif()

  target_include_directories(aegis PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${MMSOURCE112}/core
    ${MMSOURCE112}/core/sourcehook
    ${AEGIS_PROTO_GEN}
    ${HL2SDKCS2}/thirdparty/protobuf-3.21.8/src
    ${HL2SDKCS2}/public
    ${HL2SDKCS2}/public/engine
    ${HL2SDKCS2}/public/mathlib
    ${HL2SDKCS2}/public/tier0
    ${HL2SDKCS2}/public/tier1
    ${HL2SDKCS2}/public/entity2
    ${HL2SDKCS2}/public/game/server
    ${HL2SDKCS2}/public/schemasystem
    ${HL2SDKCS2}/game/shared
    ${HL2SDKCS2}/game/server
    ${HL2SDKCS2}/common
  )
  if(MSVC)
    target_compile_options(aegis PRIVATE /EHsc /TP /std:c++17 /bigobj)
  endif()
  if(WIN32)
    target_link_libraries(aegis PRIVATE
      ws2_32 bcrypt Psapi
      ${HL2SDKCS2}/lib/public/win64/tier0.lib
      ${HL2SDKCS2}/lib/public/win64/interfaces.lib
      ${HL2SDKCS2}/lib/public/win64/mathlib.lib
    )
  endif()
endif()

install(FILES addons/metamod/aegis.vdf DESTINATION addons/metamod)
