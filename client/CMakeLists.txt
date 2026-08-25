cmake_minimum_required(VERSION 3.20)
project(aegis_attest_client LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(aegis_attest_client
  src/main.cpp
  src/crypto.cpp
  src/http_client.cpp
  src/probes.cpp
  src/json_lite.cpp
)
target_include_directories(aegis_attest_client PRIVATE src)

if(WIN32)
  target_link_libraries(aegis_attest_client PRIVATE winhttp bcrypt psapi)
  target_compile_definitions(aegis_attest_client PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
