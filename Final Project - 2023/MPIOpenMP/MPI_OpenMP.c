#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include "time.h"
#include "MPIOpenMP.h"

// define the number of objects that we wanna found in the picture
// the forumla is numberOfOBjects * 3
#define MAX_VALUE 9 


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

      // Diffrence function - calculte the diffrence between picture and object 
    double diffrence(int picture, int object)
    {
        return abs((picture - object) / picture);
    }

    // Matching Tool that support the main matching function
    double matchingTool(ElementFromFile* picture,ElementFromFile* object, int startRow, int startCol)
    {
        double matching = 0;
        int dim = object->dim;
        int endRow = startRow + dim;
        int endCol = startCol + dim;
        for(int i = startRow; i < endRow; i++){
            for(int j = startCol; j < endCol; j++){
                double picVal = *(picture->matrixElement[i] + j);
                double objVal = *(object->matrixElement[i - startRow] + j - startCol);
                matching += diffrence(picVal, objVal);
            }
        }

        return matching / (dim * dim);
    }

    // Main matching picture, manage and doing the whole matching proccess, sending to the matchingTool Function, begining indexs 
    // every time.
    int matchingObjectInPicture(ElementFromFile* picture,ElementFromFile* object, double matchingValue, int* startRow, int* startCol)
    {
        const int picDim = picture->dim;
        const int ObjDim = object->dim;
        for (int i = 0, BorderRow = picDim - ObjDim; i <= BorderRow; i++) {
            for(int j = 0, BorderCol = picDim - ObjDim; j <= BorderCol; j++){
                double myMatching = matchingTool(picture, object, i, j);
                if(myMatching < matchingValue){
                    *startRow = i;
                    *startCol = j;
                    return 1;
                }
            }
        }
        return 0;
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
        int index = 0;
        for(int i = 0; i < element.dim; i++)
        {
            for(int j = 0; j < element.dim; j++)
            {
                matrixFrom2Dto1D[index++] = element.matrixElement[i][j];
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

int main(int argc, char* argv[]) {

    // send number of ojbects and not nine 
    int resultArray[MAX_VALUE] = {0}; // result Matrix - [0] - id of object [1] - first index [2] - second index and etc... 
	tags tagsStruct = {0,1,2,3,4,5,6,7}; // Define tagsStruct Values
    MPI_Status status ; // return status for receive 
    int  my_rank;  // rank of Proccess          
	int  numberOfProccess; // Number of Proccess 
	int source = 0; // source of master proccess 
    int numberOfPicsThatsSent = 0; // number of pictures that sent - "there is still work to do"
    int errorCode; // Error for send Recieve 

    double start,end; // Time


    // Tags to control working flow 
    int tagIamWorking = 7; 
    int tagNoMoreWork = 8;
 
 
    // Reading Values From File - 

    int numberOfObject;
    int currentObject;  
    double matchingValue;
    int numberOfPictures;   
	
	/* start up MPI */
	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &numberOfProccess);

	if (my_rank == 0)
	{
            int pictureId = 0; // Id of picture - reading from file
            int objectCounter = 0; // Counter of matching objects
            int numberOfProccessCurrentlyWorking = 0; // Current number of proccess that working
		   
            // Open File, if file do not open correctly - print Error, and return 0.

            FILE* filePointerInput = fopen("input.txt", "r");
            if(filePointerInput == NULL)
            {
                printf("ERROR while opening the input file\n");
                return 0;
            }

            FILE* filePointerOutput = fopen("output.txt", "w");
            if(filePointerOutput == NULL)
            {
                printf("ERROR while opening the output file\n");
                return 0;
            }
            
            // Read Matching Value
            if(!readMatchingValue(filePointerInput,&matchingValue)) 
            {
                printf("ERROR while reading matching Value\n");
                return 0;
            }
            
            // Define And Allocated Array of Pictures elements
            ElementFromFile* picture = readElement(filePointerInput,&numberOfPictures);
            if (picture == NULL)
            {
                printf("ERROR while reading/creating Array of pictures\n");
                return 0;
            }

            // Define And Allocated Array of Objects elements
            ElementFromFile* object = readElement(filePointerInput,&numberOfObject);
            if (object == NULL)
            {
                printf("ERROR while reading/creating Array of objects\n");
                return 0;
            }

            // Close input File
            fclose(filePointerInput);

            // StartTime
            //clock_t start = clock();
            start = MPI_Wtime();

            // Sending Data to other proccess
            for(int i = 1; i < numberOfProccess;i++)
            {
                errorCode = MPI_Send(&numberOfPictures,1,MPI_INT,i,tagsStruct.tagNumberOfPictures, MPI_COMM_WORLD);

                if (errorCode != MPI_SUCCESS) 
                {
                    printf("Error sending message from rank %d. Exiting...\n", my_rank);
                    MPI_Abort(MPI_COMM_WORLD, errorCode);
                }

                errorCode = MPI_Send(&numberOfObject,1,MPI_INT,i,tagsStruct.tagNumberOfObject, MPI_COMM_WORLD);

                if (errorCode != MPI_SUCCESS) 
                {
                    printf("Error sending message from rank %d. Exiting...\n", my_rank);
                    MPI_Abort(MPI_COMM_WORLD, errorCode);
                }
                errorCode = MPI_Send(&matchingValue,1,MPI_DOUBLE,i,tagsStruct.tagMatchingValue, MPI_COMM_WORLD);

                if (errorCode != MPI_SUCCESS) 
                {
                    printf("Error sending message from rank %d. Exiting...\n", my_rank);
                    MPI_Abort(MPI_COMM_WORLD, errorCode);
                }

            }

            // Sending Objects to other proccess 
            for(int k = 1; k < numberOfProccess; k++) 
            {
                for(int i = 0; i < numberOfObject; i++)
                {
                    SendingStructData(object[i],k);
                }
            }

            if(numberOfProccess < numberOfPictures)

            {

            // Sending Pictuers to proccess - each proccess get one picture
            for(int i = 1; i < numberOfProccess; i++)
                {
                    SendingStructData(picture[numberOfPicsThatsSent],i);
                    errorCode = MPI_Send(&tagIamWorking,1,MPI_INT,i,tagIamWorking, MPI_COMM_WORLD);
                    if (errorCode != MPI_SUCCESS) 
                    {
                        printf("Error sending message from rank %d. Exiting...\n", my_rank);
                        MPI_Abort(MPI_COMM_WORLD, errorCode);
                    }
                    numberOfProccessCurrentlyWorking++;
                    numberOfPicsThatsSent++;
                }

            }

            else

            {
                for(int i = 1; i <= numberOfPictures; i++)
                {
                    SendingStructData(picture[numberOfPicsThatsSent],i);
                    errorCode = MPI_Send(&tagIamWorking,1,MPI_INT,i,tagIamWorking, MPI_COMM_WORLD);
                    if (errorCode != MPI_SUCCESS) 
                    {
                        printf("Error sending message from rank %d. Exiting...\n", my_rank);
                        MPI_Abort(MPI_COMM_WORLD, errorCode);
                    }
                    numberOfProccessCurrentlyWorking++;
                    numberOfPicsThatsSent++;
                }

                for(int j = numberOfPicsThatsSent; j < numberOfProccess; j++)
                {
                    errorCode = MPI_Send(&tagNoMoreWork,1,MPI_INT,j,tagNoMoreWork, MPI_COMM_WORLD);
                    if (errorCode != MPI_SUCCESS) 
                    {
                        printf("Error sending message from rank %d. Exiting...\n", my_rank);
                        MPI_Abort(MPI_COMM_WORLD, errorCode);
                    }
                }     
                
            }
    
            do {
                // Recieve Result Data From other proccess 
                errorCode = MPI_Recv(&pictureId,1,MPI_INT, MPI_ANY_SOURCE,tagsStruct.tagPictureId,MPI_COMM_WORLD,&status);
                if (errorCode != MPI_SUCCESS)
                {
                    printf("Error receiving message in rank %d. Exiting...\n", my_rank);
                    MPI_Abort(MPI_COMM_WORLD, errorCode);
                }
                errorCode = MPI_Recv(&objectCounter,1,MPI_INT, MPI_ANY_SOURCE,tagsStruct.tagNumberOfObjectResult,MPI_COMM_WORLD,&status);
                if (errorCode != MPI_SUCCESS)
                {
                    printf("Error receiving message in rank %d. Exiting...\n", my_rank);
                    MPI_Abort(MPI_COMM_WORLD, errorCode);
                }
                errorCode = MPI_Recv(&resultArray,9,MPI_INT,MPI_ANY_SOURCE,tagsStruct.tagResultArray,MPI_COMM_WORLD,&status);
                if (errorCode != MPI_SUCCESS)
                {
                    printf("Error receiving message in rank %d. Exiting...\n", my_rank);
                    MPI_Abort(MPI_COMM_WORLD, errorCode);
                }
               
                resultArrayToFile(resultArray,filePointerOutput,pictureId,objectCounter);
                numberOfProccessCurrentlyWorking--;

                // Check if there is more pictures to check - "more work to do"
               if(numberOfPicsThatsSent < numberOfPictures)
               {
                SendingStructData(picture[numberOfPicsThatsSent],status.MPI_SOURCE);
                numberOfProccessCurrentlyWorking++;
                numberOfPicsThatsSent++;
                errorCode = MPI_Send(&tagIamWorking,1,MPI_INT,status.MPI_SOURCE,tagIamWorking, MPI_COMM_WORLD);
                if (errorCode != MPI_SUCCESS) 
                {
                    printf("Error sending message from rank %d. Exiting...\n", my_rank);
                    MPI_Abort(MPI_COMM_WORLD, errorCode);
                }

               }

                // There is not left pictures to check - "there is no more work to do"
               else
               
               {
                 SendingStructData(picture[0],status.MPI_SOURCE);
                errorCode = MPI_Send(&tagNoMoreWork,1,MPI_INT,status.MPI_SOURCE,tagNoMoreWork, MPI_COMM_WORLD);
                if (errorCode != MPI_SUCCESS) 
                    {
                        printf("Error sending message from rank %d. Exiting...\n", my_rank);
                        MPI_Abort(MPI_COMM_WORLD, errorCode);
                    } 
               }
                


            } while (numberOfProccessCurrentlyWorking > 0);

            // Close file After Writing tne results              
            fclose(filePointerOutput);
            //clock_t end = clock();
            end = MPI_Wtime();
            //double total_elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
            printf("Total calculation time for the Parllel solution is: %f \n", end-start);
            FreeElementFromFile(picture,numberOfPictures);
            FreeElementFromFile(object,numberOfObject);
            
    }

    else

    {
        // Save Indexs of matching 
        int startRow = 0;
        int startCol = 0;
        int temp = 0; // The variable `temp` is used to receive different tags and control whether there is more work to do or not by the tag. 
   
       // Recieve Data 
       
        errorCode = MPI_Recv(&numberOfObject, 1, MPI_INT, source, tagsStruct.tagNumberOfObject, MPI_COMM_WORLD, &status);
        if (errorCode != MPI_SUCCESS)
            {
                printf("Error receiving message in rank %d. Exiting...\n", my_rank);
                MPI_Abort(MPI_COMM_WORLD, errorCode);
            }
        errorCode = MPI_Recv(&matchingValue, 1, MPI_DOUBLE, source, tagsStruct.tagMatchingValue, MPI_COMM_WORLD, &status);
        if (errorCode != MPI_SUCCESS)
            {
                printf("Error receiving message in rank %d. Exiting...\n", my_rank);
                MPI_Abort(MPI_COMM_WORLD, errorCode);
            }
               
        errorCode = MPI_Recv(&numberOfPictures, 1, MPI_INT, source, tagsStruct.tagNumberOfPictures, MPI_COMM_WORLD, &status);
        if (errorCode != MPI_SUCCESS)
            {
                printf("Error receiving message in rank %d. Exiting...\n", my_rank);
                MPI_Abort(MPI_COMM_WORLD, errorCode);
            }
               
        // Each Proccess - get the whole objects
        ElementFromFile** objectsArray = (ElementFromFile**)malloc(numberOfObject*sizeof(ElementFromFile*));
        for(int i = 0; i < numberOfObject; i++)
            objectsArray[i] = RecievingStructData(source,&status);

        // Recieve Picture from proccess 0 
        ElementFromFile* picture;
        if(my_rank <= numberOfPictures)
        {
            picture = RecievingStructData(source,&status);
        }
        MPI_Recv(&temp, 1, MPI_INT, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (errorCode != MPI_SUCCESS)
            {
                printf("Error receiving message in rank %d. Exiting...\n", my_rank);
                MPI_Abort(MPI_COMM_WORLD, errorCode);
            }

          while(status.MPI_TAG == tagIamWorking)
        {
            #pragma omp parallel for
                for(int i = 0; i < numberOfObject; i++)
                {
                    if(matchingObjectInPicture(picture,objectsArray[i],matchingValue,&startRow,&startCol) == 1)
                    {
                        int index = -1;
                        #pragma omp atomic capture
                        {
                            index = currentObject;
                            currentObject += 3;
                        }
                        resultArray[index] = objectsArray[i]->id;
                        resultArray[index+1] = startRow;
                        resultArray[index+2] = startCol;
                    }
                }

            // Send Results 
            errorCode = MPI_Send(&picture->id,1,MPI_INT, 0, tagsStruct.tagPictureId, MPI_COMM_WORLD);
            if (errorCode != MPI_SUCCESS) 
            {
                printf("Error sending message from rank %d. Exiting...\n", my_rank);
                MPI_Abort(MPI_COMM_WORLD, errorCode);
            }

             errorCode = MPI_Send(&currentObject,1,MPI_INT, 0,tagsStruct.tagNumberOfObjectResult, MPI_COMM_WORLD);
            if (errorCode != MPI_SUCCESS) 
            {
                printf("Error sending message from rank %d. Exiting...\n", my_rank);
                MPI_Abort(MPI_COMM_WORLD, errorCode);
            }

            errorCode = MPI_Send(&resultArray,9,MPI_INT, 0, tagsStruct.tagResultArray, MPI_COMM_WORLD);
            if (errorCode != MPI_SUCCESS) 
            {
                printf("Error sending message from rank %d. Exiting...\n", my_rank);
                MPI_Abort(MPI_COMM_WORLD, errorCode);
            }

            currentObject = 0;
            picture = RecievingStructData(source,&status);
            errorCode = MPI_Recv(&temp,1,MPI_INT, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (errorCode != MPI_SUCCESS)
                {
                    printf("Error receiving message in rank %d. Exiting...\n", my_rank);
                    MPI_Abort(MPI_COMM_WORLD, errorCode);
                }      
        }

	}

    /* shut down MPI */
	MPI_Finalize();
	return 0;
}
        

    




