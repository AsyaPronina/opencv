project(gapi.3rdparty.vasot)

set(TARGET_NAME gapi.3rdparty.vasot)

file(GLOB_RECURSE lib_srcs *.cpp)
file(GLOB_RECURSE lib_hdrs *.hpp)

add_library(${TARGET_NAME} STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${lib_srcs} ${lib_hdrs})

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(${TARGET_NAME} PRIVATE opencv_core)
ocv_target_include_modules(${TARGET_NAME} opencv_core)

set_target_properties(${TARGET_NAME}
  PROPERTIES OUTPUT_NAME ${TARGET_NAME}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${TARGET_NAME}
  COMPILE_PDB_NAME_DEBUG "${TARGET_NAME}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${TARGET_NAME} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()
