cmake_minimum_required(VERSION 3.27)
project(jhq CXX)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT TARGET faiss)
    find_package(faiss QUIET)
    if(NOT faiss_FOUND)
        set(JHQ_FAISS_DIR "" CACHE PATH "")
        if(JHQ_FAISS_DIR)
            add_subdirectory(${JHQ_FAISS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/faiss_build)
        else()
            message(FATAL_ERROR "faiss not found; set JHQ_FAISS_DIR to the FAISS source tree")
        endif()
    endif()
endif()

add_library(jhq STATIC
    IndexJHQ.cpp
    IndexJHQ.h
    impl/IndexJHQTrain.cpp
    impl/IndexJHQSearch.cpp
    impl/IndexJHQIO.cpp
    IndexIVFJHQ.h
    IndexIVFJHQ.cpp
    impl/IndexIVFJHQTrain.cpp
    impl/IndexIVFJHQSearch.cpp
    impl/IndexIVFJHQCodec.cpp
    impl/IndexIVFJHQOptimize.cpp
    impl/IndexIVFJHQPredecode.cpp
    impl/IndexIVFJHQIO.cpp
    impl/IndexIVFJHQScanner.h
    impl/IndexIVFJHQScanner.cpp
)

target_include_directories(jhq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(jhq PUBLIC
    faiss
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
)

option(JHQ_USE_LAPACK "Enable LAPACK-based QR in JHQ rotation" ON)
include(CheckIncludeFileCXX)
if(JHQ_USE_LAPACK)
    set(JHQ_LAPACK_FOUND FALSE)

    if(APPLE)
        check_include_file_cxx(Accelerate/Accelerate.h JHQ_HAVE_ACCELERATE_H)
        if(JHQ_HAVE_ACCELERATE_H)
            target_compile_definitions(jhq PRIVATE JHQ_USE_ACCELERATE=1)
            target_link_libraries(jhq PUBLIC "-framework Accelerate")
            set(JHQ_LAPACK_FOUND TRUE)
        endif()
    endif()

    if(NOT JHQ_LAPACK_FOUND)
        find_package(LAPACK)
        check_include_file_cxx(lapacke.h JHQ_HAVE_LAPACKE_H)
        set(JHQ_LAPACKE_OK FALSE)
        if(JHQ_HAVE_LAPACKE_H)
            find_library(JHQ_LAPACKE_LIBRARY NAMES lapacke)
            if(JHQ_LAPACKE_LIBRARY)
                set(JHQ_LAPACKE_OK TRUE)
            else()
                foreach(lib ${LAPACK_LIBRARIES})
                    if(lib MATCHES "lapacke")
                        set(JHQ_LAPACKE_OK TRUE)
                    endif()
                endforeach()
            endif()
        endif()

        if(LAPACK_FOUND AND JHQ_HAVE_LAPACKE_H AND JHQ_LAPACKE_OK)
            target_compile_definitions(jhq PRIVATE JHQ_USE_LAPACK=1)
            target_link_libraries(jhq PUBLIC ${LAPACK_LIBRARIES})
            if(JHQ_LAPACKE_LIBRARY)
                target_link_libraries(jhq PUBLIC ${JHQ_LAPACKE_LIBRARY})
            endif()
            set(JHQ_LAPACK_FOUND TRUE)
        endif()
    endif()

    if(NOT JHQ_LAPACK_FOUND)
        message(WARNING "JHQ: LAPACK not found. JL rotation will fail at runtime.")
    endif()
endif()

if(UNIX)
    target_link_libraries(jhq PUBLIC pthread m dl)
endif()

target_compile_definitions(jhq PUBLIC FINTEGER=int)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(jhq PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-sign-compare
    )
endif()

include(CheckCXXSourceCompiles)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(JHQ_IS_ARM TRUE)
else()
    set(JHQ_IS_ARM FALSE)
endif()

if(JHQ_IS_ARM)
    check_cxx_source_compiles("
#include <arm_neon.h>
int main() {
    float32x4_t a = vdupq_n_f32(0.0f);
    float32x4_t b = vdupq_n_f32(1.0f);
    float32x4_t c = vaddq_f32(a, b);
    (void)c;
    return 0;
}" JHQ_HAVE_NEON)

    if(JHQ_HAVE_NEON)
        target_compile_definitions(jhq PRIVATE JHQ_HAS_NEON=1)
    endif()

    set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS "-march=armv8-a+sve")
    check_cxx_source_compiles("
#include <arm_sve.h>
int main() {
    svfloat32_t a = svdup_f32(0.0f);
    (void)a;
    return 0;
}" JHQ_HAVE_SVE)
    set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})

    if(JHQ_HAVE_SVE)
        target_compile_definitions(jhq PRIVATE JHQ_HAS_SVE=1)
    endif()
else()
    set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512vl -mavx512bw -mavx512dq")
    check_cxx_source_compiles("
#include <immintrin.h>
int main() {
    __m512 a = _mm512_setzero_ps();
    __m512 b = _mm512_setzero_ps();
    __m512 c = _mm512_add_ps(a, b);
    (void)c;
    return 0;
}" JHQ_HAVE_AVX512)
    set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})

    if(NOT JHQ_HAVE_AVX512)
        set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma")
        check_cxx_source_compiles("
#include <immintrin.h>
int main() {
    __m256 a = _mm256_setzero_ps();
    __m256 b = _mm256_setzero_ps();
    __m256 c = _mm256_fmadd_ps(a, b, b);
    (void)c;
    return 0;
}" JHQ_HAVE_AVX2)
        set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
    endif()

    if(JHQ_HAVE_AVX512)
        target_compile_options(jhq PRIVATE
            -mavx512f
            -mavx512vl
            -mavx512bw
            -mavx512dq
            -mfma
        )
        target_compile_definitions(jhq PRIVATE JHQ_HAS_AVX512=1)
    elseif(JHQ_HAVE_AVX2)
        target_compile_options(jhq PRIVATE
            -mavx2
            -mfma
        )
        target_compile_definitions(jhq PRIVATE JHQ_HAS_AVX2=1)
    endif()
endif()
