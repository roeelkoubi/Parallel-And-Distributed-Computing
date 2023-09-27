#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Sequential.h"
#include "time.h"

    // Diffrence function - calculte the diffrence between picture and object 
    double diffrence(int picture, int object)
    {
        return abs((picture - object) / picture);
    }

    double matchingTool(ElementFromFile* picture, ElementFromFile* object, int startRow, int startCol)
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
    int matchingObjectInPicture(ElementFromFile* picture, ElementFromFile* object, double matchingValue, int* startRow, int* startCol)
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

    ElementFromFile* ElementsArray = (ElementFromFile*)malloc(*numberOfElements*sizeof(ElementFromFile));
    if(!ElementsArray)
    {
        printf("ERROR while allocating Memory\n");
        fclose(filePointer);
        return NULL;
    }

    for(int i = 0; i < *numberOfElements; i++)
    {
        if (fscanf(filePointer, "%d", &ElementsArray[i].id) != 1) 
        {
            printf("ERROR while reading id of Element\n");
            fclose(filePointer);
            free(ElementsArray);  // free the previously allocated memory
            return NULL;
        }

        if (fscanf(filePointer, "%d", &ElementsArray[i].dim) != 1) 
        {
            printf("ERROR while reading Dim of Element\n");
            fclose(filePointer);
            free(ElementsArray);  // free the previously allocated memory
            return NULL;
        }

        ElementsArray[i].matrixElement = (int**)malloc(ElementsArray[i].dim*sizeof(int*));
        if(!ElementsArray[i].matrixElement)
        {
            printf("ERROR while allocating Memory\n");
            fclose(filePointer);
            // free the previously allocated memory
            for(int j = 0; j < i; j++) {
                for(int k = 0; k < ElementsArray[j].dim; k++) {
                    free(ElementsArray[j].matrixElement[k]);
                }
                free(ElementsArray[j].matrixElement);
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
                // free the previously allocated memory
                for(int j = 0; j <= i; j++) {
                    for(int k = 0; k < ElementsArray[j].dim; k++) {
                        free(ElementsArray[j].matrixElement[k]);
                    }
                    free(ElementsArray[j].matrixElement);
                }
                free(ElementsArray);
                return NULL;
            }

            for (int colum = 0; colum < ElementsArray[i].dim; colum++)
            {
                if (fscanf(filePointer, "%d", &ElementsArray[i].matrixElement[rows][colum]) != 1)
                {
                    printf("ERROR while reading\n");
                    fclose(filePointer);
                    // free the previously allocated memory
                    for(int j = 0; j <= i; j++) {
                        for(int k = 0; k < ElementsArray[j].dim; k++) {
                            free(ElementsArray[j].matrixElement[k]);
                        }
                        free(ElementsArray[j].matrixElement);
                    }
                    free(ElementsArray);
                    return NULL;
                }
            }
        }
    }

    return ElementsArray;
}

    int main(int argc, char* argv[]) {

            double matchingValue;
            int numberOfPictures;
            int numberOfObject;

            // Open File, if file do not open correctly - print Error, and return 0.
            FILE* filePointer = fopen("input.txt", "r");
            if (!filePointer)
            {
                printf("File do not open correctly\n");
                return 0;
            }

            // Read Matching Value
            if(!readMatchingValue(filePointer,&matchingValue)) 
            {
                printf("ERROR while reading matching Value\n");
                return 0;
            }
            
            // Define And Allocated Array of Pictures elements
            ElementFromFile* picture = readElement(filePointer,&numberOfPictures);
            if (picture == NULL)
            {
                printf("ERROR while reading/creating Array of pictures\n");
                return 0;
            }

            // Define And Allocated Array of Objects elements
            ElementFromFile* object = readElement(filePointer,&numberOfObject);
            if (object == NULL)
            {
                printf("ERROR while reading/creating Array of objects\n");
                return 0;
            }

            // Close File
            fclose(filePointer);

    // Initialize a 3x3 array to store the IDs and positions of found objects
    int ObjectsFound[3][3] = {{-1}};

    // Initialize a counter variable to keep track of how many objects have been found
    int counterObjects;

    // Initialize a variable to keep track of the total elapsed time for object matching
    double total_elapsed = 0;

    // Iterate over each picture
    for (int i = 0; i < numberOfPictures; i++) {

    // Reset the object counter to 0 for each picture
    counterObjects = 0;

    // Iterate over each object until three different objects are found or there are no more objects
    for (int j = 0; j < numberOfObject && counterObjects < 3; j++) {
        
        // Start a timer to measure the execution time of the matchingObjectInPicture function
        clock_t start = clock();

        // Try to match the current object with the current picture
        if (matchingObjectInPicture(&picture[i], &object[j], matchingValue, &ObjectsFound[counterObjects][1], &ObjectsFound[counterObjects][2])) {
            
            // If the object matches, add it to the list of found objects and increment the object counter
            ObjectsFound[counterObjects++][0] = object[j].id;
        }

        // Stop the timer and add the elapsed time to the total elapsed time
        clock_t end = clock();
        total_elapsed += ((double) (end - start)) / CLOCKS_PER_SEC; 
    }

    // If three different objects were found, print their positions in the picture
    if (counterObjects == 3) {
        printf("Picture %d: found Objects: %d Position(%d,%d) ; %d Position(%d,%d) ; %d  Position(%d,%d) \n", picture[i].id,
            ObjectsFound[0][0], ObjectsFound[0][1], ObjectsFound[0][2],
            ObjectsFound[1][0], ObjectsFound[1][1], ObjectsFound[1][2],
            ObjectsFound[2][0], ObjectsFound[2][1], ObjectsFound[2][2]);
    }
    // If fewer than three different objects were found, print a message saying so
    else
    {
        printf("Picture %d: No three different Objects were found  \n", picture[i].id);
    }
    }

    // Print the total elapsed time for the sequential solution
    printf("Total calculation time for the sequential solution is: %lf \n", total_elapsed);

    // Return 0 to indicate successful completion of the program
    free(picture);
    free(object);
    return 0;

}
