cmake_minimum_required(VERSION 3.24)

project(compare_fss_impls LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenSSL REQUIRED)
find_package(OpenMP)

add_executable(myl7_batch_bench myl7_batch_bench.cu)
target_include_directories(myl7_batch_bench PRIVATE
    /home/yy404nf/FSS-Work/fss/include)
target_link_libraries(myl7_batch_bench PRIVATE OpenSSL::Crypto)
target_compile_options(myl7_batch_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-maes,-msse4.2>)
if(OpenMP_CXX_FOUND)
    target_link_libraries(myl7_batch_bench PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(myl7_batch_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
endif()

add_executable(libfss_batch_bench
    libfss_batch_bench.cpp
    /home/yy404nf/FSS-Work/libfss/cpp/fss-client.cpp
    /home/yy404nf/FSS-Work/libfss/cpp/fss-server.cpp
    /home/yy404nf/FSS-Work/libfss/cpp/fss-common.cpp)
target_include_directories(libfss_batch_bench PRIVATE
    /home/yy404nf/FSS-Work/libfss/cpp)
target_compile_definitions(libfss_batch_bench PRIVATE
    AESNI
    OPENSSL_AES_H
    aesni_set_encrypt_key=AES_set_encrypt_key
    aesni_set_decrypt_key=AES_set_decrypt_key
    aesni_encrypt=AES_encrypt
    aesni_decrypt=AES_decrypt)
target_compile_options(libfss_batch_bench PRIVATE -O3 -maes -msse4.2)
target_link_libraries(libfss_batch_bench PRIVATE OpenSSL::Crypto gmpxx gmp pthread dl ssl)
