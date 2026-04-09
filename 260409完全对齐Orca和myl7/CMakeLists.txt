cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "/usr/bin/g++-13")
  set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-13" CACHE FILEPATH "CUDA host compiler")
endif()

project(orca_myl7_align LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(OpenMP)

find_library(GMP_LIBRARY gmp REQUIRED)
find_library(GMPXX_LIBRARY gmpxx REQUIRED)

set(AUTO_DETECT_CUDA_ARCH ON CACHE BOOL "Detect CUDA architectures from the current machine")

if(AUTO_DETECT_CUDA_ARCH)
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE DETECTED_GPU_CAPS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)

  if(DETECTED_GPU_CAPS)
    string(REPLACE "\n" ";" DETECTED_GPU_CAPS "${DETECTED_GPU_CAPS}")
    set(DETECTED_CUDA_ARCHES "")
    foreach(cap IN LISTS DETECTED_GPU_CAPS)
      string(STRIP "${cap}" cap)
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*$" "\\1\\2" arch "${cap}")
      if(arch)
        list(APPEND DETECTED_CUDA_ARCHES "${arch}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES DETECTED_CUDA_ARCHES)
    if(DETECTED_CUDA_ARCHES)
      set(CMAKE_CUDA_ARCHITECTURES "${DETECTED_CUDA_ARCHES}" CACHE STRING "Auto-detected CUDA architectures" FORCE)
    endif()
  endif()

  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Fallback CUDA architectures" FORCE)
  endif()
endif()

message(STATUS "Detected CUDA toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(ORCA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../FSS/Orca")
if(NOT EXISTS "${ORCA_ROOT}/gpu/gpu_mem.cu")
  set(ORCA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.deps/FSS/Orca")
endif()

set(MYL7_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../FSS/myl7")
if(NOT EXISTS "${MYL7_ROOT}/include")
  set(MYL7_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../fss")
endif()
if(NOT EXISTS "${MYL7_ROOT}/include")
  set(MYL7_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.deps/fss")
endif()

set(LIBFSS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libfss")
if(NOT EXISTS "${LIBFSS_ROOT}/cpp/fss-client.cpp")
  set(LIBFSS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.deps/libfss")
endif()

message(STATUS "Using ORCA root: ${ORCA_ROOT}")
message(STATUS "Using myl7 root: ${MYL7_ROOT}")
message(STATUS "Using libfss root: ${LIBFSS_ROOT}")

function(add_common_target target_name)
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>
    $<$<COMPILE_LANGUAGE:CUDA>:-m64>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-w,-fpermissive,-fpic,-pthread,-march=native>
  )
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()

list(GET CMAKE_CUDA_ARCHITECTURES 0 MYL7_CUDA_ARCH)

function(add_myl7_gpu_target target_name source_name common_header)
  set(output_path "${CMAKE_CURRENT_BINARY_DIR}/${target_name}")
  set(stamp_path "${output_path}.stamp")
  add_custom_command(
    OUTPUT "${stamp_path}"
    BYPRODUCTS "${output_path}"
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
      -O3
      -std=c++20
      --extended-lambda
      --generate-code=arch=compute_${MYL7_CUDA_ARCH},code=sm_${MYL7_CUDA_ARCH}
      -I"${CMAKE_CURRENT_SOURCE_DIR}"
      -I"${MYL7_ROOT}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/${source_name}"
      -o "${output_path}"
    COMMAND ${CMAKE_COMMAND} -E touch "${stamp_path}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/${source_name}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${common_header}"
    VERBATIM
  )
  add_custom_target(${target_name} ALL DEPENDS "${stamp_path}")
endfunction()

function(add_myl7_cpu_target target_name source_name common_header)
  set(output_path "${CMAKE_CURRENT_BINARY_DIR}/${target_name}")
  set(stamp_path "${output_path}.stamp")
  add_custom_command(
    OUTPUT "${stamp_path}"
    BYPRODUCTS "${output_path}"
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
      -O3
      -std=c++20
      --generate-code=arch=compute_${MYL7_CUDA_ARCH},code=sm_${MYL7_CUDA_ARCH}
      -I"${CMAKE_CURRENT_SOURCE_DIR}"
      -I"${MYL7_ROOT}/include"
      -Xcompiler=-maes,-msse4.2,-fopenmp
      "${CMAKE_CURRENT_SOURCE_DIR}/${source_name}"
      -o "${output_path}"
      -lcrypto
    COMMAND ${CMAKE_COMMAND} -E touch "${stamp_path}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/${source_name}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${common_header}"
    VERBATIM
  )
  add_custom_target(${target_name} ALL DEPENDS "${stamp_path}")
endfunction()

add_executable(orca_dcf_batch_keygen_bench
  orca_dcf_batch_keygen_bench.cu
  ${ORCA_ROOT}/gpu/gpu_mem.cu
)
target_include_directories(orca_dcf_batch_keygen_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${ORCA_ROOT}"
)
target_link_libraries(orca_dcf_batch_keygen_bench PRIVATE
  CUDA::cuda_driver
  CUDA::cudart
  CUDA::curand
)
add_common_target(orca_dcf_batch_keygen_bench)

add_executable(orca_dpf_batch_keygen_bench
  orca_dpf_batch_keygen_bench.cu
  ${ORCA_ROOT}/gpu/gpu_mem.cu
)
target_include_directories(orca_dpf_batch_keygen_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${ORCA_ROOT}"
)
target_link_libraries(orca_dpf_batch_keygen_bench PRIVATE
  CUDA::cuda_driver
  CUDA::cudart
  CUDA::curand
)
add_common_target(orca_dpf_batch_keygen_bench)

add_executable(orca_dcf_singlekey_bench
  orca_dcf_singlekey_bench.cu
  ${ORCA_ROOT}/gpu/gpu_mem.cu
)
target_include_directories(orca_dcf_singlekey_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${ORCA_ROOT}"
)
target_link_libraries(orca_dcf_singlekey_bench PRIVATE
  CUDA::cuda_driver
  CUDA::cudart
  CUDA::curand
)
add_common_target(orca_dcf_singlekey_bench)

add_executable(orca_dpf_singlekey_bench
  orca_dpf_singlekey_bench.cu
  ${ORCA_ROOT}/gpu/gpu_mem.cu
)
target_include_directories(orca_dpf_singlekey_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${ORCA_ROOT}"
)
target_link_libraries(orca_dpf_singlekey_bench PRIVATE
  CUDA::cuda_driver
  CUDA::cudart
  CUDA::curand
)
add_common_target(orca_dpf_singlekey_bench)

add_myl7_gpu_target(myl7_dcf_batch_keygen_bench myl7_dcf_batch_keygen_bench.cu dcf_bench_common.h)
add_myl7_gpu_target(myl7_dpf_batch_keygen_bench myl7_dpf_batch_keygen_bench.cu dpf_bench_common.h)
add_myl7_gpu_target(myl7_dcf_singlekey_bench myl7_dcf_singlekey_bench.cu dcf_bench_common.h)
add_myl7_gpu_target(myl7_dpf_singlekey_bench myl7_dpf_singlekey_bench.cu dpf_bench_common.h)
add_myl7_cpu_target(myl7_cpu_dcf_batch_keygen_bench myl7_cpu_dcf_batch_keygen_bench.cu dcf_bench_common.h)
add_myl7_cpu_target(myl7_cpu_dpf_batch_keygen_bench myl7_cpu_dpf_batch_keygen_bench.cu dpf_bench_common.h)
add_myl7_cpu_target(myl7_cpu_dcf_singlekey_bench myl7_cpu_dcf_singlekey_bench.cu dcf_bench_common.h)
add_myl7_cpu_target(myl7_cpu_dpf_singlekey_bench myl7_cpu_dpf_singlekey_bench.cu dpf_bench_common.h)

add_executable(libfss_dcf_batch_keygen_bench
  libfss_dcf_batch_keygen_bench.cu
  libfss_aesni_shim.cpp
  ${LIBFSS_ROOT}/cpp/fss-client.cpp
  ${LIBFSS_ROOT}/cpp/fss-common.cpp
)
target_include_directories(libfss_dcf_batch_keygen_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${LIBFSS_ROOT}/cpp"
)
target_link_libraries(libfss_dcf_batch_keygen_bench PRIVATE
  CUDA::cudart
  OpenSSL::Crypto
  ${GMPXX_LIBRARY}
  ${GMP_LIBRARY}
)
target_compile_definitions(libfss_dcf_batch_keygen_bench PRIVATE OPENSSL_AES_H)
add_common_target(libfss_dcf_batch_keygen_bench)

add_executable(libfss_dpf_batch_keygen_bench
  libfss_dpf_batch_keygen_bench.cu
  libfss_aesni_shim.cpp
  ${LIBFSS_ROOT}/cpp/fss-client.cpp
  ${LIBFSS_ROOT}/cpp/fss-common.cpp
)
target_include_directories(libfss_dpf_batch_keygen_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${LIBFSS_ROOT}/cpp"
)
target_link_libraries(libfss_dpf_batch_keygen_bench PRIVATE
  CUDA::cudart
  OpenSSL::Crypto
  ${GMPXX_LIBRARY}
  ${GMP_LIBRARY}
)
target_compile_definitions(libfss_dpf_batch_keygen_bench PRIVATE OPENSSL_AES_H)
add_common_target(libfss_dpf_batch_keygen_bench)

add_executable(libfss_dcf_singlekey_bench
  libfss_dcf_singlekey_bench.cu
  libfss_aesni_shim.cpp
  ${LIBFSS_ROOT}/cpp/fss-client.cpp
  ${LIBFSS_ROOT}/cpp/fss-common.cpp
  ${LIBFSS_ROOT}/cpp/fss-server.cpp
)
target_include_directories(libfss_dcf_singlekey_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${LIBFSS_ROOT}/cpp"
)
target_link_libraries(libfss_dcf_singlekey_bench PRIVATE
  CUDA::cudart
  OpenSSL::Crypto
  ${GMPXX_LIBRARY}
  ${GMP_LIBRARY}
)
target_compile_definitions(libfss_dcf_singlekey_bench PRIVATE OPENSSL_AES_H)
add_common_target(libfss_dcf_singlekey_bench)

add_executable(libfss_dpf_singlekey_bench
  libfss_dpf_singlekey_bench.cu
  libfss_aesni_shim.cpp
  ${LIBFSS_ROOT}/cpp/fss-client.cpp
  ${LIBFSS_ROOT}/cpp/fss-common.cpp
  ${LIBFSS_ROOT}/cpp/fss-server.cpp
)
target_include_directories(libfss_dpf_singlekey_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${LIBFSS_ROOT}/cpp"
)
target_link_libraries(libfss_dpf_singlekey_bench PRIVATE
  CUDA::cudart
  OpenSSL::Crypto
  ${GMPXX_LIBRARY}
  ${GMP_LIBRARY}
)
target_compile_definitions(libfss_dpf_singlekey_bench PRIVATE OPENSSL_AES_H)
add_common_target(libfss_dpf_singlekey_bench)
