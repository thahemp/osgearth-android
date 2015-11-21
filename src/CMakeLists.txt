#the old construct SUBDIRS( was substituded by ADD_SUBDIRECTORY that is to be preferred according on CMake docs.
SET(MODULE_USER_STATIC_OR_DYNAMIC ${OSGEARTH_USER_DEFINED_DYNAMIC_OR_STATIC})
FOREACH( lib
         osgEarth
         osgEarthAnnotation
         osgEarthFeatures
         osgEarthUtil
         osgEarthSymbology
 )

    ADD_SUBDIRECTORY(${lib})
    
    SET_PROPERTY(TARGET ${lib} PROPERTY FOLDER "Libs")

ENDFOREACH( lib )

ADD_SUBDIRECTORY( osgEarthDrivers )
ADD_SUBDIRECTORY( osgEarthExtensions )

IF(NOT OSG_BUILD_PLATFORM_IPHONE AND NOT OSG_BUILD_PLATFORM_IPHONE_SIMULATOR AND NOT ANDROID)
ADD_SUBDIRECTORY( applications )
ENDIF()

IF (Qt5Widgets_FOUND OR QT4_FOUND AND NOT ANDROID AND OSGEARTH_USE_QT)
    ADD_SUBDIRECTORY(osgEarthQt)
    SET_PROPERTY(TARGET osgEarthQt PROPERTY FOLDER "Libs")
ENDIF()

IF(MSVC80)
  OPTION(OSGEARTH_MSVC_GENERATE_PLUGINS_AND_WRAPPERS_MANIFESTS "Generate or not manifests files under VS8 for dynamically loaded dlls" ON)
ENDIF(MSVC80)

IF(ANDROID)
    configure_file("${OSG_ANDROID_TEMPLATES}/Android.mk.src.in" "${CMAKE_CURRENT_BINARY_DIR}/Android.mk")
ENDIF()
