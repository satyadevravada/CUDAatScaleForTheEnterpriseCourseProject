################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
    $(info WARNING - x86_64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=x86_64 instead)
    TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
    $(info WARNING - ARMv7 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=armv7l instead)
    TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
    $(info WARNING - aarch64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=aarch64 instead)
    TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
    $(info WARNING - ppc64le variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=ppc64le instead)
    TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
    $(info WARNING - GCC variable has been deprecated)
    $(info WARNING - please use HOST_COMPILER=$(GCC) instead)
    HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
    $(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 sbsa ppc64le armv7l))
    ifneq ($(TARGET_ARCH),$(HOST_ARCH))
        ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 sbsa ppc64le))
            TARGET_SIZE := 64
        else ifneq (,$(filter $(TARGET_ARCH),armv7l))
            TARGET_SIZE := 32
        endif
    else
        TARGET_SIZE := $(shell getconf LONG_BIT)
    endif
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif

# sbsa and aarch64 systems look similar. Need to differentiate them at host level for now.
ifeq ($(HOST_ARCH),aarch64)
    ifeq ($(CUDA_PATH)/targets/sbsa-linux,$(shell ls -1d $(CUDA_PATH)/targets/sbsa-linux 2>/dev/null))
        HOST_ARCH := sbsa
        TARGET_ARCH := sbsa
    endif
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-sbsa x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# When on native aarch64 system with userspace of 32-bit, change TARGET_ARCH to armv7l
ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_SIZE),aarch64-aarch64-32)
    TARGET_ARCH = armv7l
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifeq ($(TARGET_OS),darwin)
    ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
        HOST_COMPILER ?= clang++
    endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        ifeq ($(TARGET_OS),linux)
            HOST_COMPILER ?= arm-linux-gnueabihf-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
        else ifeq ($(TARGET_OS),android)
            HOST_COMPILER ?= arm-linux-androideabi-g++
        endif
    else ifeq ($(TARGET_ARCH),aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/q++
        else ifeq ($(TARGET_OS), android)
            HOST_COMPILER ?= aarch64-linux-android-clang++
        endif
    else ifeq ($(TARGET_ARCH),sbsa)
        HOST_COMPILER ?= aarch64-linux-gnu-g++
    else ifeq ($(TARGET_ARCH),ppc64le)
        HOST_COMPILER ?= powerpc64le-linux-gnu-g++
    endif
endif
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     :=

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib -L$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib/aarch64-linux-gnu -L$(TARGET_FS)/lib/aarch64-linux-gnu
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib -L$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/aarch64-linux-gnu -L$(TARGET_FS)/usr/lib/aarch64-linux-gnu
            LDFLAGS += --unresolved-symbols=ignore-in-shared-libs
            CCFLAGS += -isystem=$(TARGET_FS)/usr/include -I$(TARGET_FS)/usr/include -I$(TARGET_FS)/usr/include/libdrm
            CCFLAGS += -isystem=$(TARGET_FS)/usr/include/aarch64-linux-gnu -I$(TARGET_FS)/usr/include/aarch64-linux-gnu
        endif
    endif
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
        NVCCFLAGS += --qpp-config 5.4.0,gcc_ntoaarch64le
        CCFLAGS += -DWIN_INTERFACE_CUSTOM -I/usr/include/aarch64-qnx-gnu
        LDFLAGS += -lsocket
        LDFLAGS += -L/usr/lib/aarch64-qnx-gnu
        CCFLAGS += "-Wl\,-rpath-link\,/usr/lib/aarch64-qnx-gnu"
        ifdef TARGET_OVERRIDE
            LDFLAGS += -lslog2
        endif

        ifneq ($(TARGET_FS),)
            LDFLAGS += -L$(TARGET_FS)/usr/lib
            CCFLAGS += "-Wl\,-rpath-link\,$(TARGET_FS)/usr/lib"
            LDFLAGS += -L$(TARGET_FS)/usr/libnvidia
            CCFLAGS += "-Wl\,-rpath-link\,$(TARGET_FS)/usr/libnvidia"
            CCFLAGS += -I$(TARGET_FS)/../include
        endif
    endif
endif

ifdef TARGET_OVERRIDE # cuda toolkit targets override
    NVCCFLAGS += -target-dir $(TARGET_OVERRIDE)
endif

# Install directory of different arch
CUDA_INSTALL_TARGET_DIR :=
ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-gnueabihf/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),sbsa-linux)
    CUDA_INSTALL_TARGET_DIR = targets/sbsa-linux/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-android)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-android)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/ARMv7-linux-QNX/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-qnx/
else ifeq ($(TARGET_ARCH),ppc64le)
    CUDA_INSTALL_TARGET_DIR = targets/ppc64le-linux/
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

SAMPLE_ENABLED := 1

# This sample is not supported on QNX
ifeq ($(TARGET_OS),qnx)
  $(info >>> WARNING - boxFilterNPP is not supported on QNX - waiving sample <<<)
  SAMPLE_ENABLED := 0
endif

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES := -I./Common -I./Common/UtilNPP
LIBRARIES :=

################################################################################

# Gencode arguments
SMS ?=

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ifeq ($(SMS),)
# Generate PTX code from SM 35
GENCODE_FLAGS += -gencode arch=compute_35,code=compute_35
endif

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ALL_CCFLAGS += --threads 0

INCLUDES += -I../Common/UtilNPP

LIBRARIES += -lnppisu_static -lnppif_static -lnppc_static -lculibos -lfreeimage

# Attempt to compile a minimal application linked against FreeImage. If a.out exists, FreeImage is properly set up.
$(shell echo "#include \"FreeImage.h\"" > test.c; echo "int main() { return 0; }" >> test.c ; $(NVCC) $(ALL_CCFLAGS) $(INCLUDES) $(ALL_LDFLAGS) $(LIBRARIES) -l freeimage test.c)
FREEIMAGE := $(shell find a.out 2>/dev/null)
$(shell rm a.out test.c 2>/dev/null)

ifeq ("$(FREEIMAGE)","")
$(info >>> WARNING - FreeImage is not set up correctly. Please ensure FreeImage is set up correctly. <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

# Target rules
all: build

build: nppFilters

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

main.o:src/main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -std=c++17 -o $@ -c $<

nppFilters: main.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p ./bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	$(EXEC) cp $@ ./bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

run: build
	$(EXEC) ./nppFilters

clean:
	rm -f nppFilters main.o
	rm -rf ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)/boxFilterNPP

clobber: clean
