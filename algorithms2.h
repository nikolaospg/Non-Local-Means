#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <random>

/**Function with many useful functions that I used.
 * Contains:
 *  1) Simple- General functions. They were used to help me write the project:
 *      1.1) void print_vector(void* vector, int size, int flag):        prints a vector
 *      1.2) void print_vector2D(float* vector, int rows, int cols):     prints vector on 2D(as a matrix)
 *      1.3) float* random_vector(int size):                             gives us a random vector
 *      1.4) void array_free(float ** A, int m):                         frees a dynamically allocated 2D array.
 *      1.5) float* array2vector(float** A, int m, int n):               converts a 2D array to a row major vector.
 * 
 *  2) Functions for the Non Local Means Filter:
 *      2.1) float* patch_kernel(float patch_sigma, int patch_dim):                                                 Gives us the patch kernel
 *      2.2) float similarity_coefficient(float* patch1, float* patch2, float* H, float filt_sigma, int patch_dim): Gives us the similarity coefficient (WITHOUT NORMALISATION!)
 *      2.3) float* image_padding(float* image, int im_dim, int patch_dim):                                         Returns a padded version of the image.
 *      2.4) float* get_image(char* file_name, int im_dim):                                                         Gives us an array with the image values, from a .csv file.
 *      2.5) void write2file(char* file_name, float* data_vector, int vector_dims):                                 Writes an array to a file
 *      2.6) float* noise2image(float* image, int im_dim, float mean, float var):                                   Adds gaussian noise to the image
 * 
 *      2.7) float* get_patch(float* padded_image, int im_dim, int patch_dim, int corner_dim, int padded_dim, int topleft_index):
 *  Gives us an array with the patch.
 *      2.8) float pixel_filtered(float* padded_image, int im_dim, float* H, int index, int patch_dim, float filt_sigma):
 *  Gives us the values of a filtered pixel (the output of the filter)
 * 
 *      2.9) void image_normalise(float* image, int im_dim):                                     Normalises the image(the normalisation done in the beginning on the MATLAB code)
 *   
 *  
 */


/*****************************************************************************/
/*                        Simple- General functions                          */
/*****************************************************************************/


//Prints a vector (integers or floats)
//If the flag==1 then I know that I work with floats
void print_vector(void* vector, int size, int flag){

    //If the flag==1 then I know that I work with floats
    if(flag==1){
        float* vector_temp=(float*)vector;
        for(int i=0; i<size; i++){
            printf("%f  ",vector_temp[i]);
        }
    }
    //Else I know that I work with integers
    else{
        int* vector_temp=(int*)vector;
        for(int i=0; i<size; i++){
            printf("%d  ",vector_temp[i]);
        }
    }
    printf("\n\n");

}

//Prints the vector as 2D matrix
void print_vector2D(float* vector, int rows, int cols){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            printf("%f  ", vector[i*cols+j]);
        }
        printf("\n");
    }
    printf("\n");
}

//Creates a random vector with values between 0 and 1 with a certain size.
float* random_vector(int size){
    float* ret=(float*)malloc(size*sizeof(float*));
    if(ret==NULL){
        printf("Error: Couldn't allocate memory for ret on random_vector function\n");
        exit(-1);
    }
    for(int i=0; i<size; i++){
        ret[i]=rand()/(double)(RAND_MAX);
    }
    return ret;
}

//Frees a dynamically allocated array
// m is the dimension
void array_free(float ** A, int m){
    for(int i=0; i<m; i++){
        free(A[i]);
    }
    free(A);
}

//Converts a 2D array to vector (row major).
//Does not free the 2D array
float* array2vector(float** A, int m, int n){

    float* ret=(float*)malloc(m*n*sizeof(float));
    if(ret==NULL){
        printf("Error: Couldn't allocate memory for ret on array2vector function\n");
        exit(-1);
    }
    int ret_index=0;
    for(int i=0; i<m; i++){
        //each iteration of j corresponds to one element of the row
        for(int j=0; j<n; j++){
            ret[ret_index]= A[i][j];
            ret_index=ret_index+1;
        }
    }
    
    return ret;
}



/*****************************************************************************/
/*                  Functions for the Non Local Means Filter                 */
/*****************************************************************************/


/** The function below serves me by giving me the H matrix, which is used as the gaussian kernel that I apply on the patches.
 *  Inputs: 1) patch_sigma -> the sigma value of the gaussian kernel
 *          2) patch_dim -> An integer telling me the number of rows/cols that the patches have.
 *  It returns a vector (representing a row major matrix) with the values of the H.
 * */
float* patch_kernel(float patch_sigma, int patch_dim){


    /*Allocating memory and creating values useful for the computation.*/
    // I make the computation using a 2D array and then convert it to a 1D representing the same array in a row major form.
    float variance=patch_sigma*patch_sigma;
    int dim_half=patch_dim/2;                    //The dims/2 number which is used a lot later.
    float** kernel_array=(float**)malloc(patch_dim*sizeof(float*));     

    if(kernel_array==NULL){
        printf("Error: Couldn't allocate memory for kernel_array on patch_kernel function\n");
        exit(-1);
    }
    
    for(int i=0; i<patch_dim; i++){
        kernel_array[i]=(float*)malloc(patch_dim*sizeof(float));
        if(kernel_array[i]==NULL){
            printf("Error: Couldn't allocate memory for kernel_array[%d] on patch_kernel function\n",i);
            exit(-1);
        }
    }

    int row_flag;               //These two will later help me "center" the indices of the array, so that the central pixel has (0,0) coefficients.
    int col_flag;
    /*Finished with the variables and the memory allocation*/

    /*Creating the 2D array and then converting to a vector, representing the matrix in a row major form*/
    for(int i=0; i<patch_dim; i++){
        row_flag=i-dim_half;
        for(int j=0; j<patch_dim; j++){
            col_flag=j-dim_half;
            kernel_array[i][j]=exp(-(col_flag*col_flag + row_flag*row_flag)/(2*variance));        //Computing the actual value
        }
    }
    float* ret=array2vector(kernel_array, patch_dim, patch_dim);            
    array_free(kernel_array, patch_dim);
    /*Finished creating the array and converting*/
    return ret;
}

/** This function calculates the similarity coefficient that corresponds to one pair of patches. 
 *  It is not in the final weighted form, as I have to calculate all of the coefficients and then find their sum in order to normalise
 *  
 *  Inputs: 1) patch1 -> vector with the image values of the first patch
 *          2) patch2 -> vector with the image values of the second patch
 *          3) H      -> vector with the values of the patch kernel
 *          4) filt_sigma  -> the sigma value used in the matlab code (it should actually be the variance)
 *          5) patch_dim   -> An integer telling me the number of rows/cols that the patches have.
 *  It returns the float of the coefficient.
 * */
float similarity_coefficient(float* patch1, float* patch2, float* H, float filt_sigma, int patch_dim){

    float count=0;              //A counter to help me calculate our norm (squared)

    /*With the for loop I calculate the norm*/
    for(int i=0; i<patch_dim*patch_dim; i++){
        count=count+(patch1[i]*H[i]- patch2[i]*H[i]) * (patch1[i]*H[i]- patch2[i]*H[i]);
    }
    /*Finished calculating the norm squared*/

    float ret=expf(-count/filt_sigma);         //Getting the final value from the exponential
    return ret;
}

/**Function which takes as inputs the image, its dimensions and the patch dimension and returns the padded version of the image.
 *  Inputs: 1) image -> 1D array of floats represending the image.
 *          2) im_dim -> the dimension of the image
 *          3) patch_dim-> The dimension of the patches (3, 5, 7 and so on)
 *  Output is the 1D array of floats represending the padded image.
 *  The way it works is that it first mirrors the image on the left and right sides of the padded version, and once it is done the padded image(not the final form) is mirrored
 *  on the top and the bottom parts (up until then they were zero elements), and this way the final form is created.
 * */
float* image_padding(float* image, int im_dim, int patch_dim){

    /* Allocating memory and creating useful variables*/
    int corner_dim=patch_dim/2;                //The amount of elements I will have in the corners
    int padded_dim=im_dim+2*corner_dim;         //The dimension of the final, padded image.
    float* padded=(float*)calloc(padded_dim*padded_dim,sizeof(float));
    if(padded==NULL){
        printf("Error: Couldn't allocate memory for padded on image_padding function");
        exit(-1);
    }
    /*Finished with the memory and with the variables*/

    /*First I put the values of the initial image*/
    int init_index=padded_dim*corner_dim+corner_dim;  //A helpful index, indicating the exact place on the padded image where we have to put the elements, according to the situation.
    for(int i=0; i<im_dim; i++){
        for(int j=0; j<im_dim; j++){
            padded[init_index+j]=image[i*im_dim+j];
        }
        init_index=init_index+padded_dim;
    }
    /*Finished with the initial image*/

    /*Now I put the values on the left side*/
    //Each i iteration corresponds to one row
    init_index=padded_dim*corner_dim;
    for(int i=0; i<im_dim; i++){
        for(int j=0; j<corner_dim; j++){
            padded[init_index+j]=image[i*im_dim+ corner_dim- j- 1];
        }
        init_index=init_index+padded_dim;
    }
    /*Finished with the left size*/

    /*Now I put the values on the right side*/
    //Each i iteration corresponds to one row
    init_index=padded_dim*corner_dim +corner_dim + im_dim;
    for(int i=0; i<im_dim; i++){
        for(int j=0; j<corner_dim; j++){
            padded[init_index+j]=image[i*im_dim+ im_dim-1-j];
        }
        init_index=init_index+padded_dim;
    }
    /*Finished with the right size*/

    /*Now I put the values to the top side, including the two corners*/
    init_index=0;
    int init_index_padded=(corner_dim +corner_dim-1)*padded_dim;        //A second helpful index, indicating the proper element to pick from the original image vector.
    //Each i iteration corresponds to one row.
    for(int i=0; i<corner_dim; i++){
        for(int j=0; j<padded_dim; j++){
            padded[init_index+j]=padded[init_index_padded +j];
        }
        init_index=init_index+padded_dim;
        init_index_padded=init_index_padded-padded_dim;
    }
    /*Finished with the top size*/

    /*Now I put the values to the bottom side, including the two corners*/
    init_index=(corner_dim+im_dim)*padded_dim;
    init_index_padded=(corner_dim+im_dim-1)*padded_dim;
    //Each i iteration corresponds to one row.
    for(int i=0; i<corner_dim; i++){
        for(int j=0; j<padded_dim; j++){
            padded[init_index+j]=padded[init_index_padded +j];
        }
        init_index=init_index+padded_dim;
        init_index_padded=init_index_padded-padded_dim;
    }
    /*Finished with the bottom size, and the padded image is ready.*/

    return padded;
}

/**The function below serves me by giving me the values of the image in a vector representing the row major matrix. The User can get the file from
 *  the get_csv.m matlab function.
 * inputs:  1) char* file_name -> The name of the file containing the data. The file should have on image element in every row (so it is just a column with all the elements).
 *          2) im_dim -> The dimensions of the image.
 *  The output is of course the vector with the image values. Please look at my recommended steps in order to use the function.
 * */
float* get_image(char* file_name, int im_dim){
    
    /*Opening the file, initialising useful variables and allocating memory*/
    FILE* stream;
    stream=fopen(file_name,"r");
    if(stream==NULL){
        printf("Error: The image you ask for cannot be found in any file\n");
        exit(-1);
    }
    char buffer[20];
    float* image=(float*)malloc(im_dim*im_dim*(sizeof(float)));
    if(image==NULL){
        printf("Error: Couldn't allocate memory for image on get_image function");
        exit(-1);
    }
    float temp;
    /*Finished initialising and allocating*/

    /*Now reading the file*/
    //Each iteration corresponds to one row of the file, which in turn corresponds to one element of the image
    for(int i=0; i<im_dim*im_dim; i++){
        fgets(buffer, 20, stream);
        temp=atof(buffer);
        image[i]=temp;
    }
    /*Finished reading the file, and got the image values.*/
    fclose(stream);
    return image;
}

/** Function that helps me by writing a vector (in our exercise representing an image) to a file.
 *  inputs: 1)  char* name->  The name of the file where the writing is going to take place. If it does not exist, then it is created.
 *          2)  float* vector->   The vector holding the values of the image.
 *          3)  int vector_dims->   The dimension of the image
 *  A file with the data has been created. It is in the proper form to be used by the compare_results.m function, i.e. in one columm(every row has one element)
 * */
void write2file(char* file_name, float* data_vector, int vector_dims){

    /*Opening(creating) the file, creating a buffer*/
    FILE* stream;
    stream=fopen(file_name, "w");
    if(stream==NULL){
        printf("Error: Couldn't allocate memory for stream on write2file function\n");
        exit(-1);
    }
    char buffer[20];
    /*Finished with the file opening*/

    /*Writing the array on the file*/
    float temp;
    for (int i=0; i<vector_dims; i++){
        for(int j=0; j<vector_dims; j++){
            temp=data_vector[i*vector_dims+j];
            sprintf(buffer, "%f\n ", temp);
            fputs(buffer, stream);
        }
    }
    /*Finished writing on the file*/

    fclose(stream);
}

/**Function used to add gaussian noise to the image. It uses the C++ random number generator.
 *  inputs: 1)float* image -> vector with the values of our image
 *          2)int im_dim     -> the dimensions of the image
 *          3)float mean   -> the mean of the distribution of the noise added.(generally =0)
 *          4)float var    -> The variance of the noise added.
 * It returns another vector with the values of the image with the noise added.
 * */
float* noise2image(float* image, int im_dim, float mean, float var){

    /*Allocating memory, initialising variables and the generator*/
    float std=sqrt(var);
    float* image_noised=(float*)malloc(im_dim*im_dim*sizeof(float));
    if(image_noised==NULL){
        printf("Error: Couldn't allocate memory for image_noised on noise2image function");
        exit(-1);
    }

    //Creating the normal distribution variable generator, with the proper arguments:
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<float> distribution(mean,std);    
    /*Finished initialising variables and the generator*/


    /*Adding the gaussian noise to the image*/
    for(int i=0; i<im_dim*im_dim; i++){
        image_noised[i]=image[i]+distribution(generator);
    }
    /*Finished adding the noise and got the new image*/

    return image_noised;
}

/**Function used to normalise our image, as it is shown in the matlab code
 * float* image-> The image
 * int im_dim -> The dimension of the image
 * */
void image_normalise(float* image, int im_dim){

    /*First calculating the max and the min*/
    float max=0;
    for(int i=0; i<im_dim*im_dim; i++){
        if(image[i]>max){
            max=image[i];
        }
    }

    float min=1;
    for(int i=0; i<im_dim*im_dim; i++){
        if(image[i]<min){
            min=image[i];
        }
    }

    float diff=max-min;
    /*Calculated the max and the min now I am going to normalise*/
    for(int i=0; i<im_dim*im_dim; i++){
        image[i]=(image[i] -min)/diff;
    }
    /*Finished the normalisation */
}

/** Function used to get a patch from the PADDED image. 
 *  inputs: 1) float* padded_image -> The padded image
 *          2) float* patch     -> Pointer to the patch that I am going to fill.
 *          3) int patch_dim-> The dimension of the patch 
 *          4) int padded_dim      -> The dimension of the padded
 *          5) int topleft_index    -> The index(on the padded) of the topleft element of the patch
 * The function changes the patch and does not return something. */
void get_patch(float* padded_image, float* patch, int patch_dim, int padded_dim, int topleft_index){

    /*For loop used to get the patch. Each i iteration corresponds to one row */
    int help_index=topleft_index;       //This is the index of the top left element of the patch in the padded image.
    for(int i=0; i<patch_dim; i++){
        memcpy(patch+i*patch_dim, padded_image+help_index, patch_dim*sizeof(float)/sizeof(char));   //I get one line at a time using memcpy    
        help_index=help_index+padded_dim;           //This variable is changed so that it corresponds to the next line.
    }
    /*Finished computing and got the patch.*/

}

/**Function used to get the value of a pixel on the output of the filter.
 *  Inputs: 1)float* padded_image : The padded image(with the noise added)
 *          2)int im_dim          : The dimension of the non-padded image
 *          3)float* H            : The kernel we use on the patch
 *          4)int index           : The index of the pixel whose output value we want to calculate
 *          5)int patch_dim       : The dimension of the patch
 *          6)float filt_sigma    : The sigma value for the filter
 * Output: The pixel value */
float pixel_filtered(float* padded_image, int im_dim, float* H, int index, int patch_dim, float filt_sigma){

    /*Calculating useful variables and getting the patch of my pixel*/
    int corner_dim=patch_dim/2;                //The dimension of the corners on the padded image
    int padded_dim=im_dim+2*corner_dim;         //The dimension of the final, padded image.
    //I will calculate the index of the top left pixel of the patch that I want to take. I do this because the get_patch function needs this pixel:
    int initial_row1=index/im_dim;                   //This is the row in which the element is found, in the original-unpadded image.
    int initial_col1=index%im_dim;                   //This is the column in which the element is found, in the original-unpadded image.
    int top_left_index1=padded_dim*initial_row1+ initial_col1;       //This is the index of the top left element of the patch in the padded image.

    float* patch1=(float*)malloc(patch_dim*patch_dim*sizeof(float));
    if(patch1==NULL){
        printf("I could not allocate memory for patch1 in function pixel_filtered!. Exit\n");
        exit(-1);
    }
    float* patch2=(float*)malloc(patch_dim*patch_dim*sizeof(float));
    if(patch2==NULL){
        printf("I could not allocate memory for patch2 in function pixel_filtered!. Exit\n");
        exit(-1);
    }

    get_patch(padded_image, patch1, patch_dim, padded_dim, top_left_index1);
    float similarity_coef;
    float z_count=0;
    float pixel_ret=0;                  //The variable I will later return
    int initial_index=padded_dim*corner_dim+corner_dim;             //This index helps me later on finding the correct indices.
    int top_left_index2=0;                 //The top left index for the second patch(again used for the get_patch function)
    /*Finished Calculating useful variables and getting the patch of my pixel*/


    /**In this double loop, I calculate every coefficient corresponding to this patch.
     * In pixel_ret I add the coefficients multiplied by the image values.
     * The z_count serves me by giving me the total count of the coefficients.  In the end I divide the pixel_Ret by the z_count as I should.
     * */
    for(int i=0; i<im_dim; i++){
        for(int j=0; j<im_dim; j++){
            get_patch(padded_image, patch2, patch_dim, padded_dim, top_left_index2 +j);
            similarity_coef=similarity_coefficient(patch1, patch2, H, filt_sigma, patch_dim);
            z_count=z_count+similarity_coef;
            pixel_ret=pixel_ret+similarity_coef*padded_image[initial_index+j];
        }
        initial_index=initial_index+padded_dim;
        top_left_index2=top_left_index2+padded_dim;
    }
    pixel_ret=pixel_ret/z_count;
    free(patch1);
    free(patch2);
    return pixel_ret;
}

