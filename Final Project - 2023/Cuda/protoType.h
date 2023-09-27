#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

// define the number of objects that we wanna found in the picture
// the forumla is numberOfOBjects * 3
#define MAX_VALUE 9 

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





