#Makefile

OBJ = transformacion
SRC = transformacion.cu
OBJ_DEBUG = transformacion_debug

# CUDA code generation flags
ifneq ($(OS_ARCH),armv7l)
  GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
endif

GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := #-gencode arch=compute_50,code=sm_50
GENCODE_SMXX    := #-gencode arch=compute_50,code=compute_50
GENCODE_FLAGS   ?= $(GENCODE_SM30) $(GENCODE_SM32) $(GENCODE_SM35) 
$(GENCODE_SM50) $(GENCODE_SMXX)

$(OBJ):$(SRC)
	nvcc -I./includes  -O3 -m64 $(GENCODE_FLAGS)    $(SRC) -o $(OBJ)


clean: 
	rm $(OBJ) *.o *~

debug:
	nvcc -I./includes  -G -lineinfo -O3 -m64 $(GENCODE_FLAGS)    $(SRC) -o $(OBJ_DEBUG)

