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
CMAKE_BINARY_DIR = /home/qiang/software/MSF_SLAM/cmake-build-release

# Include any dependencies generated for this target.
include src/CMakeFiles/msf.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/msf.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/msf.dir/flags.make

src/CMakeFiles/msf.dir/eigenMatrix.cpp.o: src/CMakeFiles/msf.dir/flags.make
src/CMakeFiles/msf.dir/eigenMatrix.cpp.o: ../src/eigenMatrix.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qiang/software/MSF_SLAM/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/msf.dir/eigenMatrix.cpp.o"
	cd /home/qiang/software/MSF_SLAM/cmake-build-release/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/msf.dir/eigenMatrix.cpp.o -c /home/qiang/software/MSF_SLAM/src/eigenMatrix.cpp

src/CMakeFiles/msf.dir/eigenMatrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/msf.dir/eigenMatrix.cpp.i"
	cd /home/qiang/software/MSF_SLAM/cmake-build-release/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qiang/software/MSF_SLAM/src/eigenMatrix.cpp > CMakeFiles/msf.dir/eigenMatrix.cpp.i

src/CMakeFiles/msf.dir/eigenMatrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/msf.dir/eigenMatrix.cpp.s"
	cd /home/qiang/software/MSF_SLAM/cmake-build-release/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qiang/software/MSF_SLAM/src/eigenMatrix.cpp -o CMakeFiles/msf.dir/eigenMatrix.cpp.s

# Object files for target msf
msf_OBJECTS = \
"CMakeFiles/msf.dir/eigenMatrix.cpp.o"

# External object files for target msf
msf_EXTERNAL_OBJECTS =

src/msf: src/CMakeFiles/msf.dir/eigenMatrix.cpp.o
src/msf: src/CMakeFiles/msf.dir/build.make
src/msf: src/CMakeFiles/msf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qiang/software/MSF_SLAM/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable msf"
	cd /home/qiang/software/MSF_SLAM/cmake-build-release/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/msf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/msf.dir/build: src/msf

.PHONY : src/CMakeFiles/msf.dir/build

src/CMakeFiles/msf.dir/clean:
	cd /home/qiang/software/MSF_SLAM/cmake-build-release/src && $(CMAKE_COMMAND) -P CMakeFiles/msf.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/msf.dir/clean

src/CMakeFiles/msf.dir/depend:
	cd /home/qiang/software/MSF_SLAM/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qiang/software/MSF_SLAM /home/qiang/software/MSF_SLAM/src /home/qiang/software/MSF_SLAM/cmake-build-release /home/qiang/software/MSF_SLAM/cmake-build-release/src /home/qiang/software/MSF_SLAM/cmake-build-release/src/CMakeFiles/msf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/msf.dir/depend

