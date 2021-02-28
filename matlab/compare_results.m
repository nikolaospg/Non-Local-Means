%Function to be called in order to compare the MATLAB and C implementations. Please read the recommended
% steps to make the validity check. 
% After running the non local means on matlab, I use this function to take the c _denoised array (if I want to open in 
% workspace), to get the mean difference and also to print the two images.
%
% Inputs: matlab_result-> The array with the matlab denoised image (If)
%              c_filename -> The name of the file with the data of the c denoised image.
%              flag-> Flag to tell me whether to print the results or not (if flag==1 then print).
% Outputs: The c_denoised array, the difference array and the mean_diff.
% Recommened syntax: [c_denoised, mean_diff, diff_array]= compare_results(If, "denoised_image.txt", 1);


function [c_denoised, mean_diff, diff]= compare_results(matlab_result, c_filename, flag)
    c_denoised=load(c_filename);
    im_dim=sqrt(length(c_denoised));
    c_denoised=reshape(c_denoised', im_dim, im_dim);
    c_denoised=c_denoised';
    diff=c_denoised-matlab_result;
    mean_diff=mean(abs((diff(:))));
    
    if(flag==1)
        
        figure('Name', 'C denoised')
        imagesc(c_denoised); axis image;
        colormap gray;
        
        figure('Name', 'Matlab denoised')
        imagesc(matlab_result); axis image;
        colormap gray;
        
        figure('Name','Difference')
        imagesc(diff); axis image;
        colormap gray;
        
        
        
    end

end