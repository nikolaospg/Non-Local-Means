#include "algorithms2.h"
#include <cstdio>
#include <cfloat>
#define EPSILON 0.0005         //The difference threshold I chose when comparing two floats. You can change it if you want to be more strict on the checks.

#define MAX_PATCH_DIM 15        //Defining the maximum dimensoin of patches that the user could use. A bigger dimension does not make sense for a Non Local Means filter.


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

__global__ void dev_pixel_filtered(float* padded_image, float* denoized, int* im_dim, float* H, int* patch_dim, float* filt_sigma){

    /*Calculating useful variables and getting the patch of my pixel*/

    float patch1[MAX_PATCH_DIM*MAX_PATCH_DIM];      //I define these arrays this way, by thinking that there is an upper limit in the patch dimension for image denoising.
    float patch2[MAX_PATCH_DIM*MAX_PATCH_DIM];

    int corner_dim=(*patch_dim)/2;                //The dimension of the corners on the padded image
    int padded_dim=(*im_dim)+2*corner_dim;         //The dimension of the final, padded image.
    int index=blockIdx.x*blockDim.x + threadIdx.x;

    //I will calculate the index of the top left pixel of the patch that I want to take. I do this because the get_patch function needs this pixel:
    int initial_row1=index/(*im_dim);                   //This is the row in which the element is found, in the original-unpadded image.
    int initial_col1=index%(*im_dim);                   //This is the column in which the element is found, in the original-unpadded image.
    int top_left_index1=padded_dim*initial_row1+ initial_col1;       //This is the index of the top left element of the patch in the padded image.

    dev_get_patch(padded_image, patch1, *patch_dim, padded_dim, top_left_index1);
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
    
    for(int i=0; i<*im_dim; i++){
       for(int j=0; j<*im_dim; j++){
            dev_get_patch(padded_image, patch2, *patch_dim, padded_dim, top_left_index2 +j);
            similarity_coef=dev_similarity_coefficient(patch1, patch2, H, *filt_sigma, *patch_dim);
            z_count=z_count+similarity_coef;
            pixel_ret=pixel_ret+similarity_coef*padded_image[initial_index+j];
        }
        initial_index=initial_index+padded_dim;
        top_left_index2=top_left_index2+padded_dim;
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

    /*Finished working with the host*/


    printf("i am here\n");
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


    dev_pixel_filtered<<<im_dim, im_dim>>>(dev_padded, dev_denoized, dev_im_dim, dev_H, dev_patch_dim, dev_filt_sigma);
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
    cudaFree(dev_padded);
    cudaFree(dev_patch_dim);
    cudaFree(dev_im_dim);
    cudaFree(dev_filt_sigma);
    cudaFree(dev_denoized);
    cudaFree(dev_H);
    free(denoized_on_host);
    return 0;
}