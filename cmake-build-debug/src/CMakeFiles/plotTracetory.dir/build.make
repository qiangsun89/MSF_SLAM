# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/qiang/software/clion-2020.3.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/qiang/software/clion-2020.3.4/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/qiang/software/MSF_SLAM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/qiang/software/MSF_SLAM/cmake-build-debug

# Include any dependencies generated for this target.
include src/CMakeFiles/plotTracetory.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/plotTracetory.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/plotTracetory.dir/flags.make

src/CMakeFiles/plotTracetory.dir/plotTracetory.cpp.o: src/CMakeFiles/plotTracetory.dir/flags.make
src/CMakeFiles/plotTracetory.dir/plotTracetory.cpp.o: ../src/plotTracetory.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qiang/software/MSF_SLAM/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/plotTracetory.dir/plotTracetory.cpp.o"
	cd /home/qiang/software/MSF_SLAM/cmake-build-debug/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/plotTracetory.dir/plotTracetory.cpp.o -c /home/qiang/software/MSF_SLAM/src/plotTracetory.cpp

src/CMakeFiles/plotTracetory.dir/plotTracetory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/plotTracetory.dir/plotTracetory.cpp.i"
	cd /home/qiang/software/MSF_SLAM/cmake-build-debug/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qiang/software/MSF_SLAM/src/plotTracetory.cpp > CMakeFiles/plotTracetory.dir/plotTracetory.cpp.i

src/CMakeFiles/plotTracetory.dir/plotTracetory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/plotTracetory.dir/plotTracetory.cpp.s"
	cd /home/qiang/software/MSF_SLAM/cmake-build-debug/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qiang/software/MSF_SLAM/src/plotTracetory.cpp -o CMakeFiles/plotTracetory.dir/plotTracetory.cpp.s

# Object files for target plotTracetory
plotTracetory_OBJECTS = \
"CMakeFiles/plotTracetory.dir/plotTracetory.cpp.o"

# External object files for target plotTracetory
plotTracetory_EXTERNAL_OBJECTS =

src/plotTracetory: src/CMakeFiles/plotTracetory.dir/plotTracetory.cpp.o
src/plotTracetory: src/CMakeFiles/plotTracetory.dir/build.make
src/plotTracetory: /usr/local/lib/libpango_glgeometry.so
src/plotTracetory: /usr/local/lib/libpango_plot.so
src/plotTracetory: /usr/local/lib/libpango_python.so
src/plotTracetory: /usr/local/lib/libpango_scene.so
src/plotTracetory: /usr/local/lib/libpango_tools.so
src/plotTracetory: /usr/local/lib/libpango_video.so
src/plotTracetory: /usr/local/lib/libpango_geometry.so
src/plotTracetory: /usr/local/lib/libtinyobj.so
src/plotTracetory: /usr/local/lib/libpango_display.so
src/plotTracetory: /usr/local/lib/libpango_vars.so
src/plotTracetory: /usr/local/lib/libpango_windowing.so
src/plotTracetory: /usr/local/lib/libpango_opengl.so
src/plotTracetory: /usr/lib/x86_64-linux-gnu/libGLEW.so
src/plotTracetory: /usr/lib/x86_64-linux-gnu/libOpenGL.so
src/plotTracetory: /usr/lib/x86_64-linux-gnu/libGLX.so
src/plotTracetory: /usr/lib/x86_64-linux-gnu/libGLU.so
src/plotTracetory: /usr/local/lib/libpango_image.so
src/plotTracetory: /usr/local/lib/libpango_packetstream.so
src/plotTracetory: /usr/local/lib/libpango_core.so
src/plotTracetory: src/CMakeFiles/plotTracetory.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qiang/software/MSF_SLAM/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable plotTracetory"
	cd /home/qiang/software/MSF_SLAM/cmake-build-debug/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/plotTracetory.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/plotTracetory.dir/build: src/plotTracetory

.PHONY : src/CMakeFiles/plotTracetory.dir/build

src/CMakeFiles/plotTracetory.dir/clean:
	cd /home/qiang/software/MSF_SLAM/cmake-build-debug/src && $(CMAKE_COMMAND) -P CMakeFiles/plotTracetory.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/plotTracetory.dir/clean

src/CMakeFiles/plotTracetory.dir/depend:
	cd /home/qiang/software/MSF_SLAM/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qiang/software/MSF_SLAM /home/qiang/software/MSF_SLAM/src /home/qiang/software/MSF_SLAM/cmake-build-debug /home/qiang/software/MSF_SLAM/cmake-build-debug/src /home/qiang/software/MSF_SLAM/cmake-build-debug/src/CMakeFiles/plotTracetory.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/plotTracetory.dir/depend
