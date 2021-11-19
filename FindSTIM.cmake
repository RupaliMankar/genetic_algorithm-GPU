# finds the STIM library (downloads it if it isn't present)
# set STIMLIB_PATH to the directory containing the stim subdirectory (the stim repository)

include(FindPackageHandleStandardArgs)

set(STIM_ROOT $ENV{STIM_ROOT})

IF(NOT STIM_ROOT)
    MESSAGE("ERROR: STIM_ROOT environment variable must be set!")
ENDIF(NOT STIM_ROOT)

    FIND_PATH(STIM_INCLUDE_DIRS DOC "Path to STIM include directory."
              NAMES stim/image/image.h
              PATHS ${STIM_ROOT})

find_package_handle_standard_args(STIM DEFAULT_MSG STIM_INCLUDE_DIRS)
