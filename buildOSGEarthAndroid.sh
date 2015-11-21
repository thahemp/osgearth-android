#!/bin/bash

# This must match the path to your osg-android build
BASE_PATH=/home/erk/GIT-repos/osg-android

mkdir -p build && cd build

cmake .. \
-DOSG_DIR:PATH="$BASE_PATH/osg-android" \
-DOSG_INCLUDE_DIR:PATH="$BASE_PATH/include" \
-DOSG_GEN_INCLUDE_DIR:PATH="$BASE_PATH/build/include" \
-DCURL_INCLUDE_DIR:PATH="$BASE_PATH/3rdparty/curl/include" \
-DCURL_LIBRARY:PATH="$BASE_PATH/3rdparty/build/curl/obj/local/armeabi-v7a/libcurl.a" \
-DGDAL_INCLUDE_DIR:PATH="$BASE_PATH/3rdparty/gdal/include" \
-DGDAL_LIBRARY:PATH="$BASE_PATH/3rdparty/build/gdal/obj/local/armeabi-v7a/libgdal.a" \
-DGEOS_INCLUDE_DIR:PATH="$BASE_PATH/3rdparty/jni/geos-3.3.4/include" \
-DGEOS_LIBRARY:PATH="$BASE_PATH/3rdparty/build/geos/obj/local/armeabi-v7a/libgeos.a" \
-DOPENTHREADS_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libOpenThreads.a \
-DOSGDB_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libosgDB.a \
-DOSGFX_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libosgFX.a \
-DOSGGA_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libosgGA.a \
-DOSGMANIPULATOR_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libosgManipulator.a \
-DOSGSHADOW_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libosgShadow.a \
-DOSGSIM_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libosgSim.a \
-DOSGTERRAIN_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libosgTerrain.a \
-DOSGTEXT_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libosgText.a \
-DOSGUTIL_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libosgUtil.a \
-DOSGVIEWER_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libosgViewer.a \
-DOSGWIDGET_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libosgWidget.a \
-DOSG_LIBRARY=$BASE_PATH/build/obj/local/armeabi-v7a/libosg.a \
-DOSGEARTH_USE_QT:BOOL=OFF \
-DOSG_BUILD_PLATFORM_ANDROID:BOOL=ON \
-DDYNAMIC_OSGEARTH:BOOL=OFF && cp -r ../AutoGenShaders/src .

make
