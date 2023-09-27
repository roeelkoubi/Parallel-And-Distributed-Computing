#include <stdio.h>
#include <stdlib.h>
#define UNUSED(x) (void)(x)


// Struct Of element - represent element from the file (Picture or Object) and his attributes 

typedef struct 
{
    int id;
    int dim;
    int** matrixElement;     
} ElementFromFile;


// // <-- Functions -->

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
// Function to free allocated memory in the ElementsArray
void freeElementsArray(ElementFromFile* ElementsArray, int numberOfElements);










