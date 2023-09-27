#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "protoType.h"

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
void resultArrayToFile(int* resultArray, FILE* filename, int pictureId, int ObjectCounter); 
// Start Matching proccess with cuda on the GPU.
void matchingOnGPU(ElementFromFile* picture, ElementFromFile** objectsArr, int numOfObjects, double matchingValue, int* resultArray, int* rowIndex, int* colIndex, int* numOfMatchingInObject); 

int main(int argc, char* argv[]) 
{

    // send number of ojbects and not nine 
    int resultArray[MAX_VALUE] = {0}; // result Matrix - [0] - id of object [1] - first index [2] - second index and etc... 
	 tags tagsStruct = {0,1,2,3,4,5,6,7}; // Define tagsStruct Values
    MPI_Status status ; // return status for receive 
    int  my_rank;  // rank of Proccess          
	 int  numberOfProccess; // Number of Proccess 
	 int source = 0; // source of master proccess 
    int numberOfPicsThatsSent = 0; // number of pictures that sent - "there is still work to do"
    int errorCode; // Error for send Recieve 

    // Time
    double start,end; 


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
        int countNumberOfObjects = 0; // To control the nunmber of objects that we found matching in them 
   
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
               
      // Allocate memory on host
      ElementFromFile** objectsArray = (ElementFromFile**)malloc(numberOfObject*sizeof(ElementFromFile*));
      for(int i = 0; i < numberOfObject; i++) {
         objectsArray[i] = RecievingStructData(source, &status);
      }

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
            // Start Cuda Flow
            matchingOnGPU(picture, objectsArray, numberOfObject, matchingValue, resultArray, &startRow, &startCol, &countNumberOfObjects);

            // Send Results 
            errorCode = MPI_Send(&picture->id,1,MPI_INT, 0, tagsStruct.tagPictureId, MPI_COMM_WORLD);
            if (errorCode != MPI_SUCCESS) 
            {
                printf("Error sending message from rank %d. Exiting...\n", my_rank);
                MPI_Abort(MPI_COMM_WORLD, errorCode);
            }

             errorCode = MPI_Send(&countNumberOfObjects,1,MPI_INT, 0,tagsStruct.tagNumberOfObjectResult, MPI_COMM_WORLD);
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

            countNumberOfObjects = 0;
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


