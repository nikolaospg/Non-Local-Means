#include "algorithms2.h"

#include <cstdio>
#include <cfloat>

#define EPSILON 0.0005         //The difference threshold I chose when comparing two floats. You can change it if you want to be more strict on the checks.
#define MAX_PATCH_DIM 15        //Defining the maximum dimensoin of patches that the user could use. A bigger dimension does not make sense for a Non Local Means filter.

/**  The struct I use to get the data of the Final Image, with which I am going to work with on the shared implementation. 
 *  This image comes from the padded one, if we divide it into three parts. In these parts, we ALSO add a patch_dim/2 component in the borders. This is done 
 *  so that each of these parts can be indepentently loaded into the memory and the threads can get the patches correctly, otherwise the patches of the pixels 
 *  that are on the border would not be of the correct size. Look at the final_image function for more information.
 *
 *  The float* image component of the struct is the array with the data, and the int* cum_sizes component is an array which holds the sizes of each part, in a 
 *  cumulative fashion so that we can later easily access what we want*/
typedef struct{
    float* image;
    int* cum_sizes;
}FinalIm;

/** This is the function which gives me the final image, which I will later have to use to get the benefits of the shared memory. It might be a bit confusing
 *  but I did whatever I could to present it in an easily followable way.
 *  What I have to do, is divide the padded image into 9 squares(rectangles to be exact), and for each of these rectangles I want to also take some extra part, equal to 
 *  patch_size/2, from the borders so that I can later access all of the patches (if I did not take the extra part, then the border pixels would not have had their patches).
 *  After I finish getting the nine parts, I copy them, in a row major way, one by one in the larger image array (which will be returned). I also get the array which holds
 *  the sizes of the parts in a cumulative fashion, so that I can later easily access whatever parts I want.
 *
 *  inputs: float* padded_image -> The padded image.
 *          int im_dim          -> The (original) image dimension
 *          int patch_dim       -> The patch dimension
 *  outputs: final_image -> The struct of the final image (ie.e image data and the sizes).
*/
FinalIm final_image(float* padded_image, int im_dim, int patch_dim){
    int corner_dim=patch_dim/2;                //The amount of elements I will have in the corners
    int padded_dim=im_dim+2*corner_dim;         //The dimension of the final, padded image.
    int length1=padded_dim/3 +1;                //In each dimension, I have 2 parts of length equal to length1, and one part of length2
    int length2=padded_dim-2*length1;           //The length2 is the remaining part
    float* parts_array[9];                      //This array holds the pointers to the 9 image parts/
    float* image_help_pointer;                  //This will later help me access easily the parts of the image.
    int sizes[9];                                   //Holds the size of each part
    int* cum_sizes=(int*)malloc(10*sizeof(int));    //Holds the size in a cumulative way, will later be returned
    cum_sizes[0]=0;

    /*Working with the first three*/

    /*Every time I go to the new part, I first find the size of the part, by using the length1, length2 and corner_dim variables. 
     * I assign the image_help_pointer to point at the appropriate address of the padded image.
     * I allocate the memory for the specific part
     * I copy the memory, one row at a time, being very careful with the arguments 
    */

    //With the first one:
    sizes[0]=(length1+corner_dim)*(length1+corner_dim);
    image_help_pointer=padded_image;
    parts_array[0]=(float*)malloc(sizes[0]*sizeof(float));
    for(int i=0; i<length1+corner_dim; i++){
        memcpy(parts_array[0]+i*(corner_dim+length1), image_help_pointer+i*padded_dim , (corner_dim+length1)*sizeof(float));
    }
    cum_sizes[1]=sizes[0];

    //With the second one:
    sizes[1]=(length1+2*corner_dim)*(length1+corner_dim);
    image_help_pointer=padded_image +length1 -corner_dim;
    parts_array[1]=(float*)malloc(sizes[1]*sizeof(float));
    for(int i=0; i<length1+corner_dim; i++){
        memcpy(parts_array[1]+i*(2*corner_dim+length1), image_help_pointer+i*padded_dim , (2*corner_dim+length1)*sizeof(float));
    }
    cum_sizes[2]=cum_sizes[1] +sizes[1];

    //With the third one:
    sizes[2]=(length2+corner_dim)*(length1+corner_dim);
    image_help_pointer=padded_image+ 2*length1- corner_dim;
    parts_array[2]=(float*)malloc(sizes[2]*sizeof(float));
    for(int i=0; i<length1+corner_dim; i++){
        memcpy(parts_array[2]+i*(corner_dim+length2), image_help_pointer+i*padded_dim , (corner_dim+length2)*sizeof(float));
    }
    cum_sizes[3]=cum_sizes[2] +sizes[2];
    /*Finished with the first three*/


    /*Now with the three next*/
    //With the fourth one
    sizes[3]=(length1+2*corner_dim)*(length1+corner_dim);
    image_help_pointer=padded_image+ (length1-corner_dim)*padded_dim;
    parts_array[3]=(float*)malloc(sizes[3]*sizeof(float));
    for(int i=0; i<length1+2*corner_dim; i++){
        memcpy(parts_array[3]+i*(corner_dim+length1), image_help_pointer+i*padded_dim , (corner_dim+length1)*sizeof(float));
    }
    cum_sizes[4]=cum_sizes[3] +sizes[3];

    //With the fifth one
    sizes[4]=(length1+2*corner_dim)*(length1+2*corner_dim);
    image_help_pointer=padded_image+ (length1-corner_dim)*padded_dim + length1-corner_dim;
    parts_array[4]=(float*)malloc(sizes[4]*sizeof(float));
    for(int i=0; i<length1+2*corner_dim; i++){
        memcpy(parts_array[4]+i*(2*corner_dim+length1), image_help_pointer+i*padded_dim , (2*corner_dim+length1)*sizeof(float));
    }
    cum_sizes[5]=cum_sizes[4] +sizes[4];

    //With the sixth one
    sizes[5]=(length1+2*corner_dim)*(length2+corner_dim);
    image_help_pointer=padded_image+ (length1-corner_dim)*padded_dim +2*length1-corner_dim;
    parts_array[5]=(float*)malloc(sizes[5]*sizeof(float));
    for(int i=0; i<length1+2*corner_dim; i++){
        memcpy(parts_array[5]+i*(corner_dim+length2), image_help_pointer+i*padded_dim , (corner_dim+length2)*sizeof(float));
    }
    cum_sizes[6]=cum_sizes[5] +sizes[5];
    /*Finished with these three too*/

    /*Now with the three last*/
    //With the seventh
    sizes[6]=(length2+corner_dim)*(length1+corner_dim);
    image_help_pointer=padded_image+ (2*length1-corner_dim)*padded_dim ;
    parts_array[6]=(float*)malloc(sizes[6]*sizeof(float));
    for(int i=0; i<length2+corner_dim; i++){
        memcpy(parts_array[6]+i*(corner_dim+length1), image_help_pointer+i*padded_dim , (corner_dim+length1)*sizeof(float));
    }
    cum_sizes[7]=cum_sizes[6] +sizes[6];

    //With the eighth
    sizes[7]=(length1+2*corner_dim)*(length2+corner_dim);
    image_help_pointer=padded_image+ (2*length1-corner_dim)*padded_dim +length1-corner_dim;
    parts_array[7]=(float*)malloc(sizes[7]*sizeof(float));
    for(int i=0; i<length2+corner_dim; i++){
        memcpy(parts_array[7]+i*(2*corner_dim+length1), image_help_pointer+i*padded_dim , (2*corner_dim+length1)*sizeof(float));
    }
    cum_sizes[8]=cum_sizes[7] +sizes[7];

    //With the ninth
    sizes[8]=(length2+corner_dim)*(length2+corner_dim);
    image_help_pointer=padded_image+ (2*length1-corner_dim)*padded_dim +2*length1-corner_dim;
    parts_array[8]=(float*)malloc(sizes[8]*sizeof(float));
    for(int i=0; i<length2+corner_dim; i++){
        memcpy(parts_array[8]+i*(corner_dim+length2), image_help_pointer+i*padded_dim , (corner_dim+length2)*sizeof(float));
    }
    cum_sizes[9]=cum_sizes[8] +sizes[8];
    /*Finished with these ones as well*/

    //I now create the ret_image, which will be returned, and copy every part, one by one into it. 
    float* ret_image=(float*)malloc(cum_sizes[9]*sizeof(float));
    for(int i=0; i< 9; i++){
        memcpy(ret_image+cum_sizes[i], parts_array[i], sizes[i]*sizeof(float));
        free(parts_array[i]);
    }
    FinalIm ret={ret_image, cum_sizes};
    return ret;
}

/**This is the device version of the similarity_coefficient function that I had used before, for the C implementation.*/
__device__ float dev_similarity_coefficient(float* patch1, float* patch2, float* H, float filt_sigma, int patch_dim){

    //Variables useful for the computation
    float values_diff;
    float count=0;              //A counter to help me calculate our norm (squared)

    /*With the for loop I calculate the norm*/
    for(int i=0; i<patch_dim*patch_dim; i++){
        values_diff=(patch1[i]-patch2[i])*H[i];
        count=count+values_diff*values_diff;
    }
    /*Finished calculating the norm squared*/

    float ret=exp(-count/filt_sigma);         //Getting the final value from the exponential
    return ret;
}

/**This is the device version of the dev_get_patch function that I had used before, for the C implementation.*/
__device__ void dev_get_patch(float* padded_image, float* patch, int patch_dim, int padded_dim, int topleft_index){

    /*For loop used to get the patch. Each i iteration corresponds to one row */
    int help_index=topleft_index;       //This is the index of the top left element of the patch in the padded image.
    for(int i=0; i<patch_dim; i++){
        memcpy(patch+i*patch_dim, padded_image+help_index, patch_dim*sizeof(float)/sizeof(char));   //I get one line at a time using memcpy    
        help_index=help_index+padded_dim;           //This variable is changed so that it corresponds to the next line.
    }
    
    /*Got the patch.*/

}

/** The kernel I use to implement the filter. It is equivalent to the one I use for the C implementation, but here:
 *    1) I pass the denoised image as an argument, because I cannot return a float(just like I did before)
 *    2) I avoid using the malloc and free functions on the device, because of the overhead they give off. 
 *    3) I have written the functions that the pexil_filtered used in the host into __device__ forms, so that I can use them in the device.
 * */

__global__ void dev_pixel_filtered(float* padded_image, float* dev_shared_im, int* cum_sizes, float* denoized, int* im_dim, float* H, int* patch_dim, float* filt_sigma){


    /*Calculating useful variables and getting the patch of my pixel*/
    float patch1[MAX_PATCH_DIM*MAX_PATCH_DIM];          //I define these arrays this way, by thinking that there is an upper limit in the patch dimension for image denoising.
    float patch2[MAX_PATCH_DIM*MAX_PATCH_DIM];
    

    int corner_dim=(*patch_dim)/2;                //The dimension of the corners on the padded image
    int padded_dim=(*im_dim)+2*corner_dim;         //The dimension of the final, padded image.
    int index=blockIdx.x*blockDim.x + threadIdx.x;

    //Concerning the shared memory:
    extern __shared__ float shared_part[];              //The shared memory holds one of the nine parts of the large final image.
    int length1=padded_dim/3 +1;                            //These are the parameters of the size of the rectangles, look at function final_image for info
    int length2=padded_dim-2*length1;
    int shared_size=(length1+2*corner_dim)*(length1+2*corner_dim);

    //These three are arrays with valueable data I use later to control the data on the shared memory, and get my results:
    // The initial_index helps me pick the correct image value, to do the filtering.
    // The j_iterations are the number of iterations regarding the rows, that I have to do for every image part 
    // The k_iterations are the number of iterations regarding the cols.
    int initial_index[]={corner_dim*(length1 +corner_dim)+corner_dim,corner_dim*(length1+2*corner_dim) +corner_dim, corner_dim*(length2+corner_dim)+ corner_dim, corner_dim*(length1+corner_dim) +corner_dim, corner_dim*(length1+2*corner_dim)+corner_dim, corner_dim*(length2+corner_dim)+corner_dim, corner_dim*(length1+corner_dim) +corner_dim, corner_dim*(length1+2*corner_dim)+corner_dim, corner_dim*(length2+corner_dim) +corner_dim};
    int j_iterations[]={length1-corner_dim, length1-corner_dim, length1-corner_dim, length1, length1, length1, length2-corner_dim, length2-corner_dim, length2-corner_dim};
    int k_iterations[]={length1-corner_dim, length1, length2-corner_dim, length1-corner_dim, length1, length2-corner_dim,length1-corner_dim, length1, length2-corner_dim};
    

    //I will calculate the index of the top left pixel of the patch that I want to take. I do this because the get_patch function needs this pixel:
    int initial_row1=index/(*im_dim);                   //This is the row in which the element is found, in the original-unpadded image.
    int initial_col1=index%(*im_dim);                   //This is the column in which the element is found, in the original-unpadded image.
    int top_left_index1=padded_dim*initial_row1+ initial_col1;       //This is the index of the top left element of the patch in the padded image.
    dev_get_patch(padded_image, patch1, *patch_dim, padded_dim, top_left_index1);
    float similarity_coef;
    float z_count=0;
    float pixel_ret=0;                  //The variable I will later return
    float* help_ptr;
    /*Finished Calculating useful variables and getting the patch of my pixel*/

    /**In this  loop, I calculate every coefficient corresponding to this pixel.
     * In pixel_ret I add the coefficients multiplied by the image values.
     * The z_count serves me by giving me the total count of the coefficients.  In the end I divide the pixel_Ret by the z_count as I should.
     * It is repeated 9 times, to check on every one of the nine rectangles.
     * */
    
    for(int i=0; i<9; i++){
        memcpy(shared_part, dev_shared_im+cum_sizes[i], shared_size*sizeof(float));     //Filling the shared memory
        help_ptr=shared_part;
        for(int j=0; j<j_iterations[i]; j++){
            for(int k=0; k<k_iterations[i]; k++){
                dev_get_patch(help_ptr, patch2, *patch_dim, k_iterations[i]+2*corner_dim, k+ j*(k_iterations[i]+2*corner_dim) );
                similarity_coef=dev_similarity_coefficient(patch1, patch2, H, *filt_sigma, *patch_dim);
                z_count=z_count+similarity_coef;
                pixel_ret=pixel_ret+similarity_coef*help_ptr[initial_index[i] + j*(k_iterations[i]+2*corner_dim) +k];
            }
        }
        __syncthreads();
    }
    pixel_ret=pixel_ret/z_count;
    denoized[index]=pixel_ret;
}

int main(int argc, char* argv[]){ 
    srand(time(0));
    if(argc!=5){
        printf("Please pass the correct arguments\n Look at the makefile for recommended syntax\n");
    }
    /*Initialising parameters, getting useful values*/
    int im_dim=atoi(argv[1]);
    int patch_dim=atoi(argv[2]);

    //The sigma values of the kernel and of the filter
    float patch_sigma=(float)5/3;
    float filt_sigma=0.02; 

    //The parameters of the noise
    int mean=0;
    float var=0.001;
    int corner_dim=patch_dim/2;                //The amount of elements I will have in the corners
    int padded_dim=im_dim+2*corner_dim;         //The dimension of the final, padded image.
    int flag=atoi(argv[4]);                                 //Flag to judge on whether to run the denoise process on the host as well, to make a validation on the correctness of the CUDA implementation.
    int length1=padded_dim/3 +1;
    int shared_size=(length1+2*corner_dim)*(length1+2*corner_dim);              //The size of the shared memory
    /*Finished initialising parameters*/

    /*Doing the job on the host*/
    struct timespec init_host;
    clock_gettime(CLOCK_MONOTONIC, &init_host);     //Only if the flag==1 will the host results be printed


    //Checking whether the user wants a random image or not
    float* image;
    if(!strcmp(argv[3], "random")){
        image=random_vector(im_dim* im_dim);
    }
    else{
        image=get_image(argv[3], im_dim);
    }
    image_normalise(image, im_dim);
    float* image_noised=noise2image(image, im_dim, mean, var);
    float* image_padded=image_padding(image, im_dim, patch_dim);
    float* H=patch_kernel(patch_sigma, patch_dim);
    float* denoized=(float*)malloc(im_dim*im_dim*sizeof(float));
    //Doing the job on the host, if the user wants to
    if(flag==1){
        for(int i=0; i<im_dim*im_dim; i++){
            denoized[i]=pixel_filtered(image_padded, im_dim, H, i, patch_dim, filt_sigma);
        }
    }

    if(flag==1){
        struct timespec last_host;   
        clock_gettime(CLOCK_MONOTONIC, &last_host);

        long ns_host;
        int seconds_host;
        if(last_host.tv_nsec <init_host.tv_nsec){
            ns_host=init_host.tv_nsec - last_host.tv_nsec;
            seconds_host= last_host.tv_sec - init_host.tv_sec -1;
        }

        if(last_host.tv_nsec >init_host.tv_nsec){
            ns_host= last_host.tv_nsec -init_host.tv_nsec ;
            seconds_host= last_host.tv_sec - init_host.tv_sec ;
        }

        printf("For the Host time: %d,%ld\n", seconds_host, ns_host);
    }

    //The last step for the host (apart from moving the data to the device) is to get the Final image:
    FinalIm shared_im=final_image(image_padded, im_dim, patch_dim);
    /*Finished working with the host*/



    /*Doing the job on the device. Allocating what I have to and copying from the host when needed.*/
    struct timespec init_dev;
    clock_gettime(CLOCK_MONOTONIC, &init_dev);

    float* dev_padded;
    cudaMalloc((void**)&dev_padded, padded_dim*padded_dim*sizeof(float));
    if(dev_padded==NULL){
        printf("Error, couldn't allocate memory for dev_padded, exiting\n");
        exit(-1);
    }
    cudaMemcpy(dev_padded, image_padded, padded_dim*padded_dim*sizeof(float), cudaMemcpyHostToDevice);

    int* dev_patch_dim;
    cudaMalloc((void**)&dev_patch_dim, sizeof(int));
    if(dev_patch_dim==NULL){
        printf("Error, couldn't allocate memory for dev_patch_dim, exiting\n");
        exit(-1);
    }
    cudaMemcpy(dev_patch_dim, &patch_dim, sizeof(int), cudaMemcpyHostToDevice);

    int* dev_im_dim;
    cudaMalloc((void**)&dev_im_dim, sizeof(int));
    if(dev_im_dim==NULL){
        printf("Error, couldn't allocate memory for dev_im_dim, exiting\n");
        exit(-1);
    }
    cudaMemcpy(dev_im_dim, &im_dim, sizeof(int), cudaMemcpyHostToDevice);

    float* dev_filt_sigma;
    cudaMalloc((void**)&dev_filt_sigma, sizeof(float));
    if(dev_filt_sigma==NULL){
        printf("Error, couldn't allocate memory for dev_filt_sigma, exiting\n");
        exit(-1);
    }
    cudaMemcpy(dev_filt_sigma, &filt_sigma, sizeof(float), cudaMemcpyHostToDevice);

    float* dev_denoized;
    cudaMalloc((void**)&dev_denoized, im_dim*im_dim*sizeof(float));
    if(dev_denoized==NULL){
        printf("Error, couldn't allocate memory for dev_denoized, exiting\n");
        exit(-1);
    }
    
    float* dev_H;
    cudaMalloc((void**)&dev_H, patch_dim*patch_dim*sizeof(float));
    if(dev_H==NULL){
        printf("Error, couldn't allocate memory for dev_H, exiting\n");
        exit(-1);
    }
    cudaMemcpy(dev_H, H, patch_dim*patch_dim*sizeof(float), cudaMemcpyHostToDevice);

    float* dev_shared_im;
    cudaMalloc((void**)&dev_shared_im, shared_im.cum_sizes[9]*sizeof(float));
    if(dev_shared_im==NULL){
        printf("Error, couldn't allocate memory for dev_shared_im, exiting\n");
        exit(-1);
    }
    cudaMemcpy(dev_shared_im, shared_im.image, shared_im.cum_sizes[9]*sizeof(float), cudaMemcpyHostToDevice);

    int* dev_cum_sizes;
    cudaMalloc((void**)&dev_cum_sizes, 10*sizeof(int));
    if(dev_shared_im==NULL){
        printf("Error, couldn't allocate memory for dev_cum_sizes, exiting\n");
        exit(-1);
    }
    cudaMemcpy(dev_cum_sizes, shared_im.cum_sizes, 10*sizeof(int), cudaMemcpyHostToDevice);



    dev_pixel_filtered<<<im_dim, im_dim, shared_size*sizeof(float)>>>(dev_padded, dev_shared_im, dev_cum_sizes, dev_denoized, dev_im_dim, dev_H, dev_patch_dim, dev_filt_sigma);
    cudaDeviceSynchronize();
    struct timespec last_dev;   
    clock_gettime(CLOCK_MONOTONIC, &last_dev);

    long ns_dev;
    int seconds_dev;
    if(last_dev.tv_nsec <init_dev.tv_nsec){
        ns_dev=init_dev.tv_nsec - last_dev.tv_nsec;
        seconds_dev= last_dev.tv_sec - init_dev.tv_sec -1;
    }

    if(last_dev.tv_nsec >init_dev.tv_nsec){
        ns_dev= last_dev.tv_nsec -init_dev.tv_nsec ;
        seconds_dev= last_dev.tv_sec - init_dev.tv_sec ;
    }

    printf("For the Device time: %d,%ld\n", seconds_dev, ns_dev);
    /*Finished working with the device*/



    /*Moving the denoised image on the host and making a test on whether the two implementations give the same results.*/
    float* denoized_on_host=(float*)malloc(im_dim*im_dim*sizeof(float));
    if(denoized_on_host==NULL){
        printf("Error could not allocate memory for denoized_on_host, exiting\n");
        exit(-1);
    }
    cudaMemcpy(denoized_on_host, dev_denoized, im_dim*im_dim*sizeof(float), cudaMemcpyDeviceToHost);

    //Checking whether the two implementations are the same. I use a small epsilon to make the float comparisons. I print the values where there is difference                 
    if(flag==1){
        for(int i=0; i<im_dim*im_dim; i++){
            if(abs((denoized_on_host[i]- denoized[i]))>EPSILON ){
                printf("For the element with index i=%d the two denoized images are different!\n", i);
                printf("From device %f, from host %f\n", denoized_on_host[i],denoized[i]);   
            }
        }
    }
    /*Finished with the test*/

    //Freeing what I don't need
    free(denoized);
    free(image);
    free(image_noised);
    free(image_padded);
    free(H);
    free(shared_im.image);
    free(shared_im.cum_sizes);
    cudaFree(dev_cum_sizes);
    cudaFree(dev_shared_im);
    cudaFree(dev_padded);
    cudaFree(dev_patch_dim);
    cudaFree(dev_im_dim);
    cudaFree(dev_filt_sigma);
    cudaFree(dev_denoized);
    cudaFree(dev_H);
    free(denoized_on_host);
    return 0;
}