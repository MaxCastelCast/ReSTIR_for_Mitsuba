#!/bin/bash

set -e

echo "========== Building Mitsuba =========="

mkdir -p build
cd build

cmake -GNinja ..
ninja
source setpath.sh

echo "========== Entering scenes directory =========="

cd scenes

echo "========== Rendering Direct =========="

mitsuba -m scalar_rgb sceneDirect.xml -o sceneDirect.exr

echo "========== Rendering Direct RIS =========="

mitsuba -m scalar_rgb sceneDirectRIS.xml -o sceneDirectRIS.exr

echo "========== Converting to PNG =========="

convert sceneDirect.exr sceneDirect.png
convert sceneDirectRIS.exr sceneDirectRIS.png

echo "========== Done =========="