#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "protoType.h"


void handleCudaError(cudaError_t errorFromCuda, int line)
{
    if (errorFromCuda != cudaSuccess)
        {
            printf("CUDA error at line %d: %s\n", line, cudaGetErrorString(errorFromCuda));
            exit(EXIT_FAILURE);  
        }
}

int* matrix2DTo1D(ElementFromFile* element) 
{
    int* tempArray = (int*)malloc(element->dim * element->dim * sizeof(int));
    if(!tempArray)
        {
            printf("ERROR\n");
            return NULL;
        }
    int arrayIndex = 0;
    for(int i = 0; i < element->dim; i++)
        for(int j = 0; j < element->dim; j++) 
            tempArray[arrayIndex++] = element->matrixElement[i][j];
    return tempArray;
}

__global__ void matchingCuda(int* pictureArr, int* objectArr, int pictureDim, int objectDim, double matchingValue, int *rowIndex, int *colIndex, int *result) 
{
    double matching = 0.0;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if((i < pictureDim ) && (j < pictureDim ) && (j + objectDim) <= pictureDim && (i + objectDim) <= pictureDim) 
    {
        for (int k = 0; k < objectDim; k++)
        {
            for (int g = 0; g < objectDim; g++)
            {
                int pictureIndex = (i + k) * pictureDim + j + g;
                int objectIndex = k * objectDim + g;
                
                double pictureValue = pictureArr[pictureIndex];
                double objectValue = objectArr[objectIndex];
                
                double diff = pictureValue - objectValue;
                double div = pictureValue;
                
                if (div != 0.0)
                {
                    matching += abs(diff / div);
                }
            }
        }
        
        matching = matching / (objectDim * objectDim);
        
        if(matching < matchingValue)
        {
            *rowIndex = i;
            *colIndex = j;
            *result = 1;
        }
    }
}

void matchingOnGPU(ElementFromFile* picture, ElementFromFile** objectsArr, int numOfObjects, double matchingValue, int* resultArray, int* rowIndex, int* colIndex, int* numOfMatchingInObject) 
{

    // Utils

    // help us to writing the results to the array in the right locations
    int writingResultToArrayHelper = 0;

    // Cuda 
    cudaError_t errorFromCuda;

    // Object Pointers
    int* ObjectMatrixOnHost;
    int* ObjectMatrixOnDevice;

    // Picture Pointers
    int* PictureMatrixOnHost;
    int* PictureMatrixOnDevice;

    // Matching Index Pointers
    int* deviceRow;
    int* deviceCol;

    // Result Pointers
    int* resultValueOnHost;
    int* resultOnDevice;
    
    // Allocted Memory for result Value on host
    resultValueOnHost = (int*)malloc(sizeof(int));

    // Convert Matrix of picture to Array using matrix 2DTo1D function 
    PictureMatrixOnHost = matrix2DTo1D(picture);

    // PictureMatrixOnDevice = NULL;  

    // Allocted Picture Matrix on the device
    errorFromCuda = cudaMalloc((void **)&PictureMatrixOnDevice, picture->dim * picture->dim * sizeof(int));  
    handleCudaError(errorFromCuda,__LINE__);

    // Copy Picture Matrix from host to the device
    errorFromCuda = cudaMemcpy(PictureMatrixOnDevice, PictureMatrixOnHost, picture->dim * picture->dim * sizeof(int), cudaMemcpyHostToDevice);  
    handleCudaError(errorFromCuda,__LINE__);

    // Start Itrate Over the Objects - each time diffrenet object
    for(int currentObject = 0; currentObject < numOfObjects; currentObject++)
    {
        // Value for helper to found matching on the kernel
        *resultValueOnHost = 0;

        // Convert Matrix of object to Array using matrix 2DTo1D function 
        ObjectMatrixOnHost = matrix2DTo1D(objectsArr[currentObject]);

        // ObjectMatrixOnDevice = NULL;  

        // Allocted Object Matrix on the device
        errorFromCuda = cudaMalloc((void **)&ObjectMatrixOnDevice, objectsArr[currentObject]->dim * objectsArr[currentObject]->dim * sizeof(int));  
        handleCudaError(errorFromCuda,__LINE__);


        // Copy Object Matrix from host to the device
        errorFromCuda = cudaMemcpy(ObjectMatrixOnDevice, ObjectMatrixOnHost, objectsArr[currentObject]->dim * objectsArr[currentObject]->dim * sizeof(int), cudaMemcpyHostToDevice);  
        handleCudaError(errorFromCuda,__LINE__);

        // deviceRow = NULL;  

        // Allocted memory for the Row result (for matching) on the device
        errorFromCuda = cudaMalloc((void **)&deviceRow, sizeof(int));  
        handleCudaError(errorFromCuda,__LINE__);

        // deviceCol = NULL;  

        // Allocted memory for the Col result (for matching) on the device
        errorFromCuda = cudaMalloc((void **)&deviceCol, sizeof(int));  
        handleCudaError(errorFromCuda,__LINE__);

        // resultOnDevice = NULL; 

        // Allocted Result - on the device
        errorFromCuda = cudaMalloc((void **)&resultOnDevice, sizeof(int));  
        handleCudaError(errorFromCuda,__LINE__);

        // Copy result value from host to the device
        errorFromCuda = cudaMemcpy(resultOnDevice, resultValueOnHost, sizeof(int), cudaMemcpyHostToDevice);  
        handleCudaError(errorFromCuda,__LINE__);


        // ------ Kernel -----------
        
        // Define Number of threads of each block 
        int numberOfTheradsOnEachBlock = 16;

        // Define number of blocks for each grid
        int blocksOnGrid = (picture->dim + numberOfTheradsOnEachBlock - 1) / numberOfTheradsOnEachBlock; 

        // DimGrid
        dim3 dimGrid(blocksOnGrid, blocksOnGrid);

        // Dim Block
        dim3 dimBlock(numberOfTheradsOnEachBlock, numberOfTheradsOnEachBlock);

        // Lunch Kernal
        matchingCuda<<<dimGrid, dimBlock>>>(PictureMatrixOnDevice, ObjectMatrixOnDevice, picture->dim, objectsArr[currentObject]->dim,matchingValue,deviceRow,deviceCol, resultOnDevice); 
        errorFromCuda = cudaGetLastError();
        handleCudaError(errorFromCuda,__LINE__);

        // ------ Kernel -----------


        // ---- Results ----- 

        // Get The result value back from the Kernel - copy from the device to host 
        errorFromCuda = cudaMemcpy(resultValueOnHost, resultOnDevice, sizeof(int), cudaMemcpyDeviceToHost);
        handleCudaError(errorFromCuda,__LINE__);

        // If we found matching we changing the value to 1, to start manage the result flow
        if(*resultValueOnHost == 1) 
        {
            // Copy The row from device to host - this is the row index we found matching
            errorFromCuda = cudaMemcpy(rowIndex, deviceRow, sizeof(int), cudaMemcpyDeviceToHost);
            handleCudaError(errorFromCuda,__LINE__);
            // Copy The col from device to host - this is the col index we found matching
            errorFromCuda = cudaMemcpy(colIndex, deviceCol, sizeof(int), cudaMemcpyDeviceToHost);
            handleCudaError(errorFromCuda,__LINE__);

            // Assign Results for the result Array 
            resultArray[writingResultToArrayHelper] = objectsArr[currentObject]->id;
            resultArray[writingResultToArrayHelper+1] = *rowIndex;
            resultArray[writingResultToArrayHelper+2] = *colIndex;

            // Object counter
            *numOfMatchingInObject += 3;
            writingResultToArrayHelper = writingResultToArrayHelper + 3;
        }

        // Free Memory That We Allcoate 
        cudaFree(ObjectMatrixOnDevice);
        cudaFree(deviceRow);
        cudaFree(deviceCol);
        cudaFree(resultOnDevice);
        free(ObjectMatrixOnHost);

    }

    // Free Memory That We Allcoate 
    free(resultValueOnHost);
    cudaFree(PictureMatrixOnDevice);
    free(PictureMatrixOnHost);
}

