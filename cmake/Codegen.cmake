if (DEFINED ENV{PYTORCH_PYTHON})
  message(STATUS "Using python found in $ENV{PYTORCH_PYTHON}")
  set(PYCMD "$ENV{PYTORCH_PYTHON}")
else()
  SET(PYCMD "python")
endif()

# ---[ Write the macros file
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/../caffe2/core/macros.h.in
    ${CMAKE_BINARY_DIR}/caffe2/core/macros.h)

# ---[ Installing the header files
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../caffe2
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_BINARY_DIR}/caffe2/core/macros.h
        DESTINATION include/caffe2/core)

# ---[ ATen specific
# SET_SOURCE_FILES_PROPERTIES must be in the same CMakeLists.txt file as the target that includes the file
# so we need to set these commands here rather than in src/TH
if (NOT ANDROID AND NOT IOS)
  IF(C_SSE4_1_FOUND AND C_SSE4_2_FOUND)
    IF(MSVC)
      SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_LIST_DIR}/../aten/src/TH/generic/simd/convolve5x5_sse.cpp PROPERTIES COMPILE_FLAGS "${MSVC_OPT_FLAG}/fp:fast")
    ELSE(MSVC)
      SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_LIST_DIR}/../aten/src/TH/generic/simd/convolve5x5_sse.cpp PROPERTIES COMPILE_FLAGS "-O3 -ffast-math")
    ENDIF(MSVC)
  ENDIF(C_SSE4_1_FOUND AND C_SSE4_2_FOUND)
  IF(C_AVX_FOUND)
    IF(MSVC)
      SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_LIST_DIR}/../aten/src/TH/generic/simd/convolve5x5_avx.cpp PROPERTIES COMPILE_FLAGS "${MSVC_OPT_FLAG}/fp:fast ${CXX_AVX_FLAGS}")
      SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_LIST_DIR}/../aten/src/TH/vector/AVX.cpp PROPERTIES COMPILE_FLAGS "${MSVC_OPT_FLAG}/arch:AVX ${CXX_AVX_FLAGS}")
    ELSE(MSVC)
      SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_LIST_DIR}/../aten/src/TH/generic/simd/convolve5x5_avx.cpp PROPERTIES COMPILE_FLAGS "-O3 -ffast-math ${CXX_AVX_FLAGS}")
      SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_LIST_DIR}/../aten/src/TH/vector/AVX.cpp PROPERTIES COMPILE_FLAGS "-O3 ${CXX_AVX_FLAGS}")
    ENDIF(MSVC)
  ENDIF(C_AVX_FOUND)
  
  IF(C_AVX2_FOUND)
    IF(MSVC)
      SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_LIST_DIR}/../aten/src/TH/vector/AVX2.cpp PROPERTIES COMPILE_FLAGS "${MSVC_OPT_FLAG}/arch:AVX2 ${CXX_AVX2_FLAGS}")
    ELSE(MSVC)
      SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_LIST_DIR}/../aten/src/TH/vector/AVX2.cpp PROPERTIES COMPILE_FLAGS "-O3 ${CXX_AVX2_FLAGS}")
    ENDIF(MSVC)
  ENDIF(C_AVX2_FOUND)
  
  IF(NOT MSVC AND NOT "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
    SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_LIST_DIR}/../aten/src/TH/THAllocator.cpp PROPERTIES COMPILE_FLAGS "-fno-openmp")
  ENDIF()
  
  FILE(GLOB cpu_kernel_cpp_in "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/cpu/*.cpp")
  
  IF(MSVC AND NOT "${CMAKE_BUILD_TYPE}" MATCHES "Debug")
    SET(MSVC_OPT_FLAG "/Ox /fp:strict ")
    SET(VCOMP_LIB "vcomp")
  ELSE()
    SET(MSVC_OPT_FLAG " ")
    SET(VCOMP_LIB "vcompd")
  ENDIF()
  
  LIST(APPEND CPU_CAPABILITY_NAMES "DEFAULT")
  IF(MSVC)
    LIST(APPEND CPU_CAPABILITY_FLAGS "${MSVC_OPT_FLAG}")
  ELSE(MSVC)
    LIST(APPEND CPU_CAPABILITY_FLAGS "-O3")
  ENDIF(MSVC)
  
  IF(CXX_AVX_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX_CPU_DEFINITION")
    LIST(APPEND CPU_CAPABILITY_NAMES "AVX")
    IF(MSVC)
      LIST(APPEND CPU_CAPABILITY_FLAGS "${MSVC_OPT_FLAG}/arch:AVX")
    ELSE(MSVC)
      LIST(APPEND CPU_CAPABILITY_FLAGS "-O3 -mavx")
    ENDIF(MSVC)
  ENDIF(CXX_AVX_FOUND)
  
  IF(CXX_AVX2_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX2_CPU_DEFINITION")
    LIST(APPEND CPU_CAPABILITY_NAMES "AVX2")
    IF(MSVC)
      LIST(APPEND CPU_CAPABILITY_FLAGS "${MSVC_OPT_FLAG}/arch:AVX2")
    ELSE(MSVC)
      LIST(APPEND CPU_CAPABILITY_FLAGS "-O3 -mavx2")
    ENDIF(MSVC)
  ENDIF(CXX_AVX2_FOUND)
  
  list(LENGTH CPU_CAPABILITY_NAMES NUM_CPU_CAPABILITY_NAMES)
  math(EXPR NUM_CPU_CAPABILITY_NAMES "${NUM_CPU_CAPABILITY_NAMES}-1")
  
  FOREACH(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
    FOREACH(IMPL ${cpu_kernel_cpp_in})
      string(REPLACE "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/" "" NAME ${IMPL})
      LIST(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
      SET(NEW_IMPL ${CMAKE_BINARY_DIR}/aten/src/ATen/${NAME}.${CPU_CAPABILITY}.cpp)
      CONFIGURE_FILE(${IMPL} ${NEW_IMPL} COPYONLY)
      SET(cpu_kernel_cpp ${NEW_IMPL} ${cpu_kernel_cpp}) # Create list of copies
      LIST(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
      IF(MSVC)
        SET(MACRO_FLAG "/DCPU_CAPABILITY=${CPU_CAPABILITY} /DCPU_CAPABILITY_${CPU_CAPABILITY}")
      ELSE(MSVC)
        SET(MACRO_FLAG "-DCPU_CAPABILITY=${CPU_CAPABILITY} -DCPU_CAPABILITY_${CPU_CAPABILITY}")
      ENDIF(MSVC)
      SET_SOURCE_FILES_PROPERTIES(${NEW_IMPL} PROPERTIES COMPILE_FLAGS "${FLAGS} ${MACRO_FLAG}")
    ENDFOREACH()
  ENDFOREACH()
  list(APPEND ATen_CPU_SRCS ${cpu_kernel_cpp})
  
  set(cwrap_files
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/Declarations.cwrap
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/THNN/generic/THNN.h
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/THCUNN/generic/THCUNN.h
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/nn.yaml
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/native_functions.yaml)
  
  FILE(GLOB all_python "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/*.py")
  
  SET(GEN_COMMAND
      ${PYCMD} ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/gen.py
      --source-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen
      --install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen
      ${cwrap_files}
  )
  
  EXECUTE_PROCESS(
      COMMAND ${GEN_COMMAND}
        --output-dependencies ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_cpp.txt
        --install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen
      RESULT_VARIABLE RETURN_VALUE
  )
  if (NOT RETURN_VALUE EQUAL 0)
      message(STATUS ${generated_cpp})
      message(FATAL_ERROR "Failed to get generated_cpp list")
  endif()
  file(READ ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_cpp.txt generated_cpp)
  file(READ ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_cpp.txt-cuda cuda_generated_cpp)
  
  file(GLOB_RECURSE all_templates "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/templates/*")
  
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/aten/src/ATen)
  
  add_custom_command(OUTPUT ${generated_cpp} ${cuda_generated_cpp}
    COMMAND ${GEN_COMMAND}
      --install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen
    DEPENDS ${all_python} ${all_templates} ${cwrap_files})
  
  # Generated headers used from a CUDA (.cu) file are
  # not tracked correctly in CMake. We make the libATen.so depend explicitly
  # on building the generated ATen files to workaround.
  add_custom_target(ATEN_CPU_FILES_GEN_TARGET DEPENDS ${generated_cpp})
  add_custom_target(ATEN_CUDA_FILES_GEN_TARGET DEPENDS ${cuda_generated_cpp})
  add_library(ATEN_CPU_FILES_GEN_LIB INTERFACE)
  add_library(ATEN_CUDA_FILES_GEN_LIB INTERFACE)
  add_dependencies(ATEN_CPU_FILES_GEN_LIB ATEN_CPU_FILES_GEN_TARGET)
  add_dependencies(ATEN_CUDA_FILES_GEN_LIB ATEN_CUDA_FILES_GEN_TARGET)
endif()
