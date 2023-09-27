#include "protoType.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include "time.h"
#include "mpi.h"

// Read the First Value from the File (Matching Value)
int readMatchingValue(FILE* filePointer,double* matchingValue);
// Read Element From File can be picture/object and create an array of structs thats describe the Element 
ElementFromFile* readElement(FILE* filePointer, int* numberOfElements);
//Sending Struct Data to other proccess
void SendingStructData(ElementFromFile element,int dest);
// Reciving Struct Data to other proccess
ElementFromFile* RecievingStructData(int source, MPI_Status* status);
// Free Element
void FreeElementFromFile(ElementFromFile* element,int numberOfElments);
// Print Result Matrix to File 
void resultArray(int* resultArray, FILE* filename, int pictureId, int ObjectCounter); 

    // Read the First Value from the File (Matching Value)
    int readMatchingValue(FILE* filePointer,double* matchingValue)
    {
            if (fscanf(filePointer, "%lf", matchingValue) != 1)
            {	
                
                printf("ERROR while reading\n");
                fclose(filePointer);
                return 0;
            }	

            return 1;
    }

    // Read Element From File can be picture/object and create an array of structs thats describe the Element 
    ElementFromFile* readElement(FILE* filePointer, int* numberOfElements)
    {
        if (fscanf(filePointer, "%d", numberOfElements) != 1) 
            {
                printf("ERROR while reading\n");
                fclose(filePointer);
                return NULL;
            }

        // Define And allocated array of Elements
        ElementFromFile* ElementsArray = (ElementFromFile*)malloc(*numberOfElements*sizeof(ElementFromFile));
        if(!ElementsArray)
        {
            printf("ERROR while allocating Memory\n");
            fclose(filePointer);
            return NULL;
        }

        // Reading id of elements
        for(int i = 0; i < *numberOfElements; i++)
        {
            if (fscanf(filePointer, "%d", &ElementsArray[i].id) != 1) 
            {
                printf("ERROR while reading id of Element\n");
                free(ElementsArray);
                fclose(filePointer);
                return NULL;
            }

            // Reading dim of elements
            if (fscanf(filePointer, "%d", &ElementsArray[i].dim) != 1) 
            {
                printf("ERROR while reading Dim of Element\n");
                free(ElementsArray);
                fclose(filePointer);
                return NULL;
            }

            // Define And allocated Matrix Element
            ElementsArray[i].matrixElement = (int**)malloc(ElementsArray[i].dim*sizeof(int*));
            if(!ElementsArray[i].matrixElement)
            {
                printf("ERROR while allocating Memory\n");
                fclose(filePointer);
                for(int k = 0; k < i; i++)
                {
                    free(ElementsArray[k].matrixElement);
                }
                free(ElementsArray);
                return NULL;
            }
            
            for (int rows = 0; rows < ElementsArray[i].dim; rows++) 
            {
                ElementsArray[i].matrixElement[rows] = (int*)malloc(ElementsArray[i].dim*sizeof(int));
                if(!ElementsArray[i].matrixElement[rows])
                {
                        printf("ERROR while allocating Memory\n");
                        fclose(filePointer);
                        for(int elementIndex = 0; elementIndex < i; elementIndex++)
                        {
                            for (int k = 0; k < rows; k++) 
                            {
                                free(ElementsArray[i].matrixElement[k]);
                            }
                        }

                        for(int k = 0; k<i; i++)
                        {
                            free(ElementsArray[k].matrixElement);
                        }
                        
                        free(ElementsArray);
                        return NULL;
                }
                        
                // Reading Data into to the Element Matrix
                for (int colum = 0; colum < ElementsArray[i].dim; colum++)
                {
                    if (fscanf(filePointer, "%d", &ElementsArray[i].matrixElement[rows][colum]) != 1)
                    {
                        printf("ERROR while reading\n");
                        fclose(filePointer);
                        for(int elementIndex = 0; elementIndex < i; elementIndex++)
                        {
                            for (int k = 0; k < ElementsArray[elementIndex].dim; k++) 
                            {
                                free(ElementsArray[i].matrixElement[k]);
                            }
                        }

                        for(int row = 0; row < i; i++)
                        {
                            free(ElementsArray[i].matrixElement);
                        }

                        free(ElementsArray);
                        return NULL;
                    }
                }
            }
        }

        return ElementsArray;
    }

    // Genral Function to send a data of an element
    void SendingStructData(ElementFromFile element,int dest)
    {
        int tagElementId = 0;
        int tagElementDim = 1;
        int tagElementMatrix = 2;

        int errorCode;

        errorCode = MPI_Send(&element.id, 1,MPI_INT, dest, tagElementId, MPI_COMM_WORLD);
        if (errorCode != MPI_SUCCESS)
        {
        printf("Error sending message\n");
        MPI_Abort(MPI_COMM_WORLD, errorCode);
        }
        errorCode = MPI_Send(&element.dim, 1,MPI_INT, dest, tagElementDim, MPI_COMM_WORLD);
        if (errorCode != MPI_SUCCESS)
        {
        printf("Error sending message\n");
        MPI_Abort(MPI_COMM_WORLD, errorCode);
        }
        int* matrixFrom2Dto1D = (int*)malloc(element.dim*element.dim*sizeof(int));
        if(!matrixFrom2Dto1D)
            {
                printf("ERROR while allocating Memory\n");
                exit(0);
            }
        //int index = 0;
        #pragma omp parallel for
        for(int i = 0; i < element.dim; i++)
        {
            for(int j = 0; j < element.dim; j++)
            {
                matrixFrom2Dto1D[i * element.dim + j] = element.matrixElement[i][j];
            }
        }

        errorCode = MPI_Send(matrixFrom2Dto1D,element.dim*element.dim, MPI_INT, dest, tagElementMatrix, MPI_COMM_WORLD);
        if (errorCode != MPI_SUCCESS)
        {
        printf("Error sending message\n");
        MPI_Abort(MPI_COMM_WORLD, errorCode);
        }
        free(matrixFrom2Dto1D);
    }

    // Genral Function to recieve a data from an element and create new element struct
    ElementFromFile* RecievingStructData(int source, MPI_Status* status)
    {
        int tagElementId = 0;
        int tagElementDim = 1;
        int tagElementMatrix = 2;

        int errorCode;

        ElementFromFile* receivedElement = (ElementFromFile*)malloc(sizeof(ElementFromFile));
        errorCode = MPI_Recv(&receivedElement->id, 1, MPI_INT, source,tagElementId, MPI_COMM_WORLD, status);
        if (errorCode != MPI_SUCCESS)
        {
            printf("Error receiving message. Exiting...\n");
            MPI_Abort(MPI_COMM_WORLD, errorCode);
        }
        errorCode = MPI_Recv(&receivedElement->dim, 1, MPI_INT, source, tagElementDim, MPI_COMM_WORLD, status);

        int* receivedElement_matrix = (int*)malloc(receivedElement->dim * receivedElement->dim * sizeof(int));
        if(!receivedElement_matrix)
            {
                printf("ERROR while allocating Memory\n");
                exit(0);
            }
        errorCode = MPI_Recv(receivedElement_matrix, receivedElement->dim * receivedElement->dim, MPI_INT, source, tagElementMatrix, MPI_COMM_WORLD, status);
        if (errorCode != MPI_SUCCESS)
        {
            printf("Error receiving message. Exiting...\n");
            MPI_Abort(MPI_COMM_WORLD, errorCode);
        }
        // Back To 2D matrix
        receivedElement->matrixElement = (int**)malloc(receivedElement->dim * sizeof(int*));
           if(!receivedElement->matrixElement)
            {
                free(receivedElement_matrix);
                printf("ERROR while allocating Memory\n");
                exit(0);
            }
        for (int i = 0; i < receivedElement->dim; i++) {
               receivedElement->matrixElement[i] = (int*)malloc(receivedElement->dim * sizeof(int));
               if(! receivedElement->matrixElement[i])
            {
                printf("ERROR while allocating Memory\n");
                for(int k = 0; k < i; k++)
                {
                    free(receivedElement->matrixElement[k]);
                }
                free(receivedElement->matrixElement);
                free(receivedElement_matrix);
                exit(0);
            }
            #pragma omp parallel for
            for (int j = 0; j < receivedElement->dim; j++) {
                 receivedElement->matrixElement[i][j] = receivedElement_matrix[i * receivedElement->dim + j];
            }
        }

        free(receivedElement_matrix);
        return receivedElement;
    }

    void resultArrayToFile(int* resultArray, FILE* filename, int pictureId, int ObjectCounter) 
    {
        if (ObjectCounter == 9)
        {
            int result = fprintf(filename, "Picture %d: found Objects: %d Position(%d,%d) ; %d Position(%d,%d) ; %d Position(%d,%d) \n", pictureId,
                resultArray[0], resultArray[1], resultArray[2],
                resultArray[3], resultArray[4], resultArray[5],
                resultArray[6], resultArray[7], resultArray[8]);
            if (result < 0)
            {
                perror("Error writing to file");
                exit(EXIT_FAILURE);
            }
        } 

        else
        
        {
            int result = fprintf(filename, "Picture %d: No three different Objects were found  \n", pictureId);
            if (result < 0) {
                perror("Error writing to file");
                exit(EXIT_FAILURE);
            }
        }
    }

    void FreeElementFromFile(ElementFromFile* elementToFree,int numberOfElments)
    {
        for(int elementFree = 0; elementFree < numberOfElments; elementFree++)
            {
                for(int rows = 0; rows < elementToFree[elementFree].dim; rows++)
                {
                    
                    free(elementToFree[elementFree].matrixElement[rows]);
                    
                }

            free(elementToFree[elementFree].matrixElement);

            }

        free(elementToFree);
    }