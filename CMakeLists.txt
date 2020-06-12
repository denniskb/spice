cmake_minimum_required(VERSION 3.0.2)

project(spice LANGUAGES CXX CUDA)

if(MSVC)
	set(CMAKE_CXX_FLAGS "/std:c++17 /EHsc /W4 /WX /fp:fast")
endif()
set(CMAKE_CUDA_FLAGS "--expt-extended-lambda -gencode=arch=compute_61,code=sm_61 --use_fast_math")

set(with_samples TRUE CACHE BOOL "Build samples (optional, recommended)")
set(with_release_asserts TRUE CACHE BOOL "Use for dev/testing. Turn off for bench/production.")
set(with_bench FALSE CACHE BOOL "Build benchmarks (optional, intended for developers, requires google benchmark)")	
set(with_tests FALSE CACHE BOOL "Build unit tests (optional, intended for developers, requires google test)")

if(with_release_asserts)
	add_definitions(-DSPICE_ASSERT_RELEASE)
endif()

# spice
file(GLOB_RECURSE spice_src spice/*)
add_library(spice ${spice_src})

target_include_directories(spice PUBLIC .)

if(MSVC)
	source_group("" FILES ${spice_src})
	source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/spice FILES ${spice_src})
endif()


# spice_test
if(with_tests)
	file(GLOB_RECURSE spice_test_src spice_test/*)
	add_executable(spice_test ${spice_test_src})

	set(gtest_inc_dir "" CACHE PATH "googletest include directory")
	set(gtest_lib_release "" CACHE FILEPATH "googletest library release version")
	set(gtest_lib_debug "" CACHE FILEPATH "googletest library debug version")

	target_include_directories(spice_test PUBLIC .)
	target_include_directories(spice_test PUBLIC ${gtest_inc_dir})
	target_link_libraries(spice_test spice optimized ${gtest_lib_release} debug ${gtest_lib_debug})

	if(MSVC)
		source_group("" FILES ${spice_test_src})
		source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/spice_test FILES ${spice_test_src})
	endif()
endif()


# spice_bench
if(with_bench)
	file(GLOB_RECURSE spice_bench_src spice_bench/*)
	add_executable(spice_bench ${spice_bench_src})

	set(gbench_inc_dir "" CACHE PATH "googlebenchmark include directory")
	set(gbench_lib_release "" CACHE FILEPATH "googlebenchmark library release version")
	set(gbench_lib_debug "" CACHE FILEPATH "googlebenchmark library debug version")

	target_include_directories(spice_bench PUBLIC .)
	target_include_directories(spice_bench PUBLIC ${gbench_inc_dir})
	target_link_libraries(spice_bench spice optimized ${gbench_lib_release} debug ${gbench_lib_debug})

	if(MSVC)
		target_link_libraries(spice_bench shlwapi.lib)
		source_group("" FILES ${spice_bench_src})
		source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/spice_bench FILES ${spice_bench_src})
	endif()
endif()


# samples
if (with_samples)
	add_executable(samples "samples/main.cpp")
	target_link_libraries(samples spice)
	
	if(MSVC)
		source_group("" FILES "samples/main.cpp")
		source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/samples FILES "samples/main.cpp")
	endif()
endif()