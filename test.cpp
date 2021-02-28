#include <complex.h>
#include "algorithms2.h"

//The struct to hold the data of the similarity statistics:
typedef struct{
    double mean;
    double rmse;
}DenoiseResult;

/** This is the function that calculates the mean and the rmse of the difference between the denoised image and the first one. It is done so that the user can actually
 * get these statistics and make a judgement on whether the denoise algorithm actually did the job and got rid of the noise (validating the algorithm).
 *
 *   Inputs: 1)float* image  ->  The image (original)
 *           2)float* denoised -> The denoised version
 *           3)int im_dim->      The dimension of the image.
 *    Outputs: The DenoiseResult containing the statistics.
 *   NOTE If you just try this function while denoising a random image, the statistics might be very small, which might seem that the denoising is very succesful, 
 *   but there is no meaning in trying to denoise a random image.*/ 

DenoiseResult compute_stats(float* image, float* denoised, int im_dim){

    double diff_mean;
    double rmse;

    /*First calculating the mean of the difference*/
    double count=0;
    for(int i=0; i<im_dim*im_dim; i++){
        count=count+ denoised[i]-image[i];
    }
    diff_mean=count/(im_dim*im_dim);
    /*Finished with the mean*/

    /*Now to calculate the rmse */
    count=0;
    double error;
    for(int i=0; i<im_dim*im_dim; i++){
        error=denoised[i]-image[i];
        count=count+ error*error;
    }
    count=count/(im_dim*im_dim);        //This is the MSE estimate.
    rmse=sqrt(count);                   //This is the RMSE estimate.
    /*Finished with the rmse as well and got my statistics*/
    DenoiseResult ret={diff_mean, rmse};
    return ret;

}

/*Steps to do the validity check*/ 
//1) I choose an image (black and white, and with equal dimensions MxM), and I get the .csv file using the get_csv.m function 
//2) I run this image on C (with the correct CL argument) and with the help of the write2file function the C result is written on a file (standard name: "denoised_image.txt")
//3) I run the matlab script, with the exact same parameters
//4) I run the compare_results.m function, with arguments the If array of the matlab script and the name of the c file. It returns the c_denoised array and the 
// mean difference of the two images. I can also open the two images (matlab anc C denoised) and look at the values 

//Recommended: Use the house.mat file to make a validity check for ease (it is easier to run it on the matlab script).
//Also When making the comparisons, just run the matlab script and then use the compare_results.m with the command window (it is easier and faster).
/*These were the simple steps for the validity check*/

/*Steps to print the Images*/
//1) We set the proper parameters and run this program. The program gives us the .txt files
//2) We pass the .txt files on the print_results.m MATLAB function.
/*The images are now printed and we can see the effect of the filter and our results*/

//I have included some ready .csv files, in case the user wants to run some images quickly.


int main(int argc, char* argv[]){ 

    srand(time(0));
    if(argc!=4){
        printf("Please pass the correct arguments\n Look at the makefile for recommended syntax\n");
    }
    /*Initialising parameters, getting useful values*/
    int im_dim=atoi(argv[1]);
    int patch_dim=atoi(argv[2]);       

    //The sigma values of the kernel and of the filter:
    float patch_sigma=(float)5/3;
    float filt_sigma=0.02; 


    //The parameters of the (gaussian) noise:
    int mean=0;
    float var=0.001;

    //Some commonly used variables
    int corner_dim=patch_dim/2;                //The amount of elements I will have in the corners
    int padded_dim=im_dim+2*corner_dim;         //The dimension of the final, padded image.
    /*Finished initialising parameters*/


    /* Getting the image and making an changes needed for the algorithm to work (normalisation, noise, padding)*/
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
    float* image_padded=image_padding(image_noised, im_dim, patch_dim);
    float* H=patch_kernel(patch_sigma, patch_dim);
    /*Got the image and the kernel*/


    /*Running the pixel_filtered function for every pixel and counting time*/
    //Starting the timer:
    struct timespec init;
    clock_gettime(CLOCK_MONOTONIC, &init);

    //Running the filter algorithm, for every pixel separately:
    float* denoised=(float*)malloc(im_dim*im_dim*sizeof(float));
    for(int i=0; i<im_dim*im_dim; i++){
        denoised[i]=pixel_filtered(image_padded, im_dim, H, i, patch_dim, filt_sigma);
    }

    //Finishing the timer:
    struct timespec last;   
    clock_gettime(CLOCK_MONOTONIC, &last);
    long ns;
    int seconds;
    //Calculating the time:
    if(last.tv_nsec <init.tv_nsec){
        ns=init.tv_nsec - last.tv_nsec;
        seconds= last.tv_sec - init.tv_sec -1;
    }
    if(last.tv_nsec >init.tv_nsec){
        ns= last.tv_nsec -init.tv_nsec ;
        seconds= last.tv_sec - init.tv_sec ;
    }
    printf("For im_dim=%d and patch_dim=%d time: %d,%ld secs.\n",im_dim, patch_dim, seconds, ns); 
    /*Finished with the algorithm and the timer*/

    /*Printing the statistics from the difference of the images. */
    DenoiseResult a=compute_stats(image, denoised, im_dim);
    printf("The algorithm validation statistics are :mean=%f and the rmse=%f\n", a.mean, a.rmse);
    /*Finished with the validity steps */

    /*Writing the results on files, in case the user wants to print them:*/
    write2file("image.txt", image, im_dim);
    write2file("noised_image.txt", image_noised , im_dim);
    write2file("denoised_image.txt", denoised , im_dim);
    /*Finished writing on files. The user can change the names or the path if he/she wants to.*/


    free(denoised);
    free(image);
    free(image_noised);
    free(image_padded);
    free(H);
    return 0;
}