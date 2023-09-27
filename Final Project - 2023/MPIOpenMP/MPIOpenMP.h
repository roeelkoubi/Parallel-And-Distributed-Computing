#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "omp.h"


// Struct Of element - represent element from the file (Picture or Object) and his attributes 
typedef struct {
    int id;
    int dim;
    int** matrixElement;
} ElementFromFile;

// Struct Of Tags 
typedef struct {
    int tagNumberOfObject;
    int tagMatchingValue;
    int tagObject;
    int tagPicture;
    int tagResultArray;
    int tagPictureId;
    int tagNumberOfObjectResult; 
    int tagNumberOfPictures;   
} tags;


// Read the First Value from the File (Matching Value)
int readMatchingValue(FILE* filePointer,double* matchingValue);
// Read Element From File can be picture/object and create an array of structs thats describe the Element 
ElementFromFile* readElement(FILE* filePointer, int* numberOfElements);
// Diffrence Function 
double diffrence(int picture, int object);
// Matching Tool that support the main matching function
double matchingTool(ElementFromFile* picture, ElementFromFile* object,int startR,int startC);
// Main Matching Function
int matchingObjectInPicture(ElementFromFile* picture, ElementFromFile* object, double matchingValue, int* startRow, int* startCol);
// Sending Struct Data to other proccess
void SendingStructData(ElementFromFile element,int dest);
// Reciving Struct Data to other proccess
ElementFromFile* RecievingStructData(int source, MPI_Status* status);
// Free Element
void FreeElementFromFile(ElementFromFile* element,int numberOfElments);
// Print Result to file
void resultArrayToFile(int* resultArray, FILE* filename, int pictureId, int ObjectCounter);



