cmake_minimum_required(VERSION 3.6)

project(neonrvm
    VERSION 0.1.0
    DESCRIPTION "Machine learning library"
    LANGUAGES C)

# Required for common installation directories
include(GNUInstallDirs)

# Find CBLAS/LAPACKE libraries
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
message(STATUS "BLAS Libraries Found: ${BLAS_LIBRARIES}")
message(STATUS "LAPACK Libraries Found: ${LAPACK_LIBRARIES}")

# Workaround for Netlib providing separate BLAS/CBLAS and LAPACK/LAPACKE libraries
foreach(BLASLIB ${BLAS_LIBRARIES})
    string(REPLACE "libblas" "libcblas" CBLASLIB ${BLASLIB})
    list(REMOVE_ITEM BLAS_LIBRARIES ${BLASLIB})
    list(APPEND BLAS_LIBRARIES ${CBLASLIB})
endforeach(BLASLIB)

foreach(LAPACKLIB ${LAPACK_LIBRARIES})
    string(REPLACE "liblapack" "liblapacke" LAPACKELIB ${LAPACKLIB})
    string(REPLACE "libblas" "libcblas" LAPACKELIB ${LAPACKELIB})
    list(REMOVE_ITEM LAPACK_LIBRARIES ${LAPACKLIB})
    list(APPEND LAPACK_LIBRARIES ${LAPACKELIB})
endforeach(LAPACKLIB)

# Enable C99 standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED 99)

# Position independent code generation
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build neonrvm library
add_library(${PROJECT_NAME} SHARED "neonrvm.c")
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# Build example program
add_executable(example "example.c")

if(WIN32)
    target_link_libraries(example ${PROJECT_NAME})
else()
    target_link_libraries(example ${PROJECT_NAME} m)
endif()

# Install neonrvm
install(FILES neonrvm.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES neonrvm-config.cmake DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/licenses/${PROJECT_NAME}")

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT ${PROJECT_NAME} DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# Package neonrvm
include(InstallRequiredSystemLibraries)
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VENDOR ${PROJECT_NAME})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
include (CPack)
